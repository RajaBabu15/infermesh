"""Worker registry loaded from YAML; API keys resolved from environment."""
from __future__ import annotations

import os
import re

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

    def get_prefill_workers(self, model: str) -> list[WorkerConfig]:
        return [w for w in self.get_workers(model) if w.role in ("prefill", "mixed")]

    def get_decode_workers(self, model: str) -> list[WorkerConfig]:
        return [w for w in self.get_workers(model) if w.role in ("decode", "mixed")]

    def all_workers(self) -> list[WorkerConfig]:
        return list(self._workers.values())

    def all_models(self) -> list[str]:
        return list({w.model for w in self._workers.values()})

    def update_utilization(self, worker_id: str, utilization: float) -> None:
        if worker_id in self._workers:
            self._workers[worker_id].cache_utilization = utilization


def parse_gpu_cache_utilization(metrics_text: str) -> float:
    """Extract vllm:gpu_cache_usage_perc from Prometheus text-format metrics."""
    pattern = re.compile(r'^vllm:gpu_cache_usage_perc\{[^}]*\}\s+([\d.eE+-]+)', re.MULTILINE)
    match = pattern.search(metrics_text)
    if match:
        return float(match.group(1))
    return 0.0
