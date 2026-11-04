"""Retry admission as a float credit ledger.

Each client request deposits credits and each retry withdraws
`rho = rho_base * (1 - alpha * hit_ratio)`, where hit_ratio is the share of the
prompt the fallback worker already holds in its prefix cache. With the defaults
(alpha = 0, rho_base = 1) this is the SRE-book rule: retries stay under about
`deposit` (10 percent) of requests.

Modes: `off` admits every retry; `static` deposits a fixed amount per request;
`headroom` deposits `quota_rps / (rho_base * arrival_rate) - 1` (floored at 0), so
no retries are admitted once load exceeds the shared upstream quota.

Envoy's minimum retry concurrency is kept as `initial_credits`, a burst reserve:
a concurrency floor inside a rate ledger would let sequential retries bypass it.
"""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from infermesh import metrics

if TYPE_CHECKING:
    from infermesh.settings import GatewayConfig

Mode = Literal["off", "static", "headroom"]


class RetryBudget:
    def __init__(
        self,
        mode: Mode = "static",
        deposit: float = 0.10,
        initial_credits: float = 3.0,
        max_credits: float = 10.0,
        max_deposit: float = 1.0,
        rho_base: float = 1.0,
        alpha: float = 0.0,
        quota_rps: float = 0.0,
        window_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.mode = mode
        self.deposit = deposit
        self.max_credits = max_credits
        self.max_deposit = max_deposit
        self.rho_base = rho_base
        self.alpha = alpha
        self.quota_rps = quota_rps
        self.window_s = window_s
        self.credits = min(initial_credits, max_credits)
        self._clock = clock
        self._arrivals: deque[float] = deque()
        self._publish()

    @classmethod
    def from_config(cls, config: "GatewayConfig") -> "RetryBudget":
        return cls(
            mode=config.retry_budget_mode,
            deposit=config.retry_budget_percent,
            initial_credits=config.retry_burst_credits,
            max_credits=config.retry_max_credits,
            rho_base=config.retry_rho_base,
            alpha=config.retry_alpha,
            quota_rps=config.retry_quota_rps,
            window_s=config.retry_rate_window_s,
        )

    def arrival_rate(self) -> float | None:
        """Requests per second over the last `window_s` seconds.

        None until at least one second of arrivals has been seen: a rate
        estimated from a fraction of a second is too low and would make
        headroom look larger than it is.
        """
        now = self._clock()
        while self._arrivals and now - self._arrivals[0] > self.window_s:
            self._arrivals.popleft()
        if not self._arrivals:
            return None
        elapsed = min(self.window_s, now - self._arrivals[0])
        if elapsed < 1.0:
            return None
        return len(self._arrivals) / elapsed

    def _deposit_now(self) -> float:
        if self.mode == "headroom" and self.quota_rps > 0:
            lam = self.arrival_rate()
            if lam is None:
                return 0.0                  # cannot prove headroom yet
            b = self.quota_rps / (self.rho_base * lam) - 1.0
            return max(0.0, min(b, self.max_deposit))
        return self.deposit

    def rho(self, hit_ratio: float = 0.0) -> float:
        """Cost of one retry to a worker holding `hit_ratio` of the prompt."""
        hit = min(max(hit_ratio, 0.0), 1.0)
        return max(self.rho_base * (1.0 - self.alpha * hit), 1e-6)

    def on_request(self) -> None:
        """A client request arrived: deposit credits."""
        if self.mode == "off":
            return
        if self.mode == "headroom":                  # only headroom mode reads the arrival rate
            self._arrivals.append(self._clock())
        self.credits = min(self.max_credits, self.credits + self._deposit_now())
        self._publish()

    def try_acquire(self, hit_ratio: float = 0.0) -> bool:
        """Admit a retry to a worker with this prefix hit ratio, or refuse."""
        if self.mode == "off":
            return True
        cost = self.rho(hit_ratio)
        if self.credits + 1e-12 < cost:
            return False
        self.credits = max(0.0, self.credits - cost)
        self._publish()
        return True

    def _publish(self) -> None:
        metrics.retry_credits.set(self.credits)
