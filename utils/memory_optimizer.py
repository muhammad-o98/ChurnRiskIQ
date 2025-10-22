"""Lightweight memory monitoring and cleanup."""
from __future__ import annotations

import gc
from typing import Any

import psutil

MEMORY_THRESHOLD_MB = 512


def get_memory_usage_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def optimize_memory():
    if get_memory_usage_mb() > MEMORY_THRESHOLD_MB:
        gc.collect()
        # Future: clear old cache entries
