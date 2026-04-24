from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def episodic_accuracy(logits, labels) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return 0.0 if total == 0 else correct / total


def mean_confidence_interval(acc_list: Iterable[float]) -> Tuple[float, float]:
    acc = np.array(list(acc_list), dtype=np.float32)
    if acc.size == 0:
        return 0.0, 0.0
    mean = float(acc.mean())
    if acc.size == 1:
        return mean, 0.0
    std = float(acc.std(ddof=1))
    ci95 = 1.96 * std / math.sqrt(acc.size)
    return mean, ci95

