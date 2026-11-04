"""Worker registry loaded from YAML; API keys resolved from environment."""
from __future__ import annotations

import os
import re
import time

import yaml

from infermesh.settings import WorkerConfig


class WorkerRegistry:
    def __init__(self, config_path: str) -> None:
        self._workers: dict[str, WorkerConfig] = {}
        self._load(config_path)

    def _load(self, path: str) -> None:
        with open(path) as f:
            data = yaml.safe_load(f)

        workers = data.get("workers", [])
        for raw in workers:
            worker = WorkerConfig(**raw)
            worker.api_key = os.getenv(worker.api_key_env, "")
            self._workers[worker.id] = worker

    def get_workers(self, model: str) -> list[WorkerConfig]:
        return [w for w in self._workers.values() if w.model == model]

    def all_workers(self) -> list[WorkerConfig]:
        return list(self._workers.values())

    def all_models(self) -> list[str]:
        return list({w.model for w in self._workers.values()})

    def update_utilization(self, worker_id: str, utilization: float) -> None:
        self.update_metrics(worker_id, utilization, None)

    def update_metrics(
        self,
        worker_id: str,
        utilization: float | None,
        queue_depth: int | None,
        now: float | None = None,
    ) -> None:
        """Record a successful engine poll. A metric the engine did not report is
        left as it was, so an absent gauge never overwrites real data with 0."""
        worker = self._workers.get(worker_id)
        if worker is None:
            return
        if utilization is not None:
            worker.cache_utilization = min(max(utilization, 0.0), 1.0)
        if queue_depth is not None:
            worker.queue_depth = max(0, queue_depth)
        worker.cache_utilization_at = time.monotonic() if now is None else now


def parse_gpu_cache_utilization(metrics_text: str) -> float | None:
    """Extract KV cache usage from vLLM Prometheus text-format metrics.

    vLLM >= 0.9.2 exports vllm:kv_cache_usage_perc; the old
    vllm:gpu_cache_usage_perc name was removed in 0.12.0. Returns None when the
    gauge is absent: unknown, not 0.0 (idle).
    """
    pattern = re.compile(
        r'^vllm:(?:kv|gpu)_cache_usage_perc(?:\{[^}]*\})?\s+([\d.eE+-]+)', re.MULTILINE
    )
    match = pattern.search(metrics_text)
    return float(match.group(1)) if match else None


def parse_num_requests_waiting(metrics_text: str) -> int | None:
    """Extract the engine's waiting-queue depth (vllm:num_requests_waiting)."""
    match = re.search(
        r'^vllm:num_requests_waiting(?:\{[^}]*\})?\s+([\d.eE+-]+)', metrics_text, re.MULTILINE
    )
    return int(float(match.group(1))) if match else None
