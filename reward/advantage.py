# -*- coding: utf-8 -*-
# Copyright 2026 Yiming. Licensed under the Apache License, Version 2.0.
"""
Pure-python reward / advantage helpers (no torch / omegaconf dependency, so
they can be unit-tested on CPU, see tests/test_advantage.py).

Step reward (dual dimension)
    r = gt_ratio + syn_weight * syn_ratio            in [0, 1 + syn_weight]
    truncated generations (hit max_gen_length) get r = 0.

Step-level advantage
    For one task, the N trajectories form a group. At step index s each
    trajectory contributes r[s]; a trajectory that already stopped "holds" its
    last reward. Advantages are the group z-score at that step:

        A(i, s) = (r_i[s] - mean_s) / std_s

    Only steps that actually exist (s < len(r_i)) become training samples, and
    zero-advantage samples are dropped (no gradient signal).
"""
from typing import Any, Dict, List, Sequence, Tuple


def z_score_normalize(lst: Sequence[float], eps: float = 1e-8) -> List[float]:
    if not lst:
        return []
    mean = sum(lst) / len(lst)
    std = (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5
    if std < eps:
        return [0.0 for _ in lst]
    return [(x - mean) / std for x in lst]


def population_std(lst: Sequence[float]) -> float:
    if not lst:
        return 0.0
    m = sum(lst) / len(lst)
    return (sum((x - m) ** 2 for x in lst) / len(lst)) ** 0.5


def step_reward(step: Dict[str, Any], syn_weight: float, max_gen_length: int) -> float:
    """Dual-dimension reward of one agent step."""
    if max_gen_length > 0 and int(step.get("response_length", 0)) >= max_gen_length - 5:
        return 0.0
    return float(step.get("gt_ratio", 0.0)) + syn_weight * float(step.get("syn_ratio", 0.0))


def step_level_advantages(rewards: Sequence[Sequence[float]], eps: float = 1e-8) -> List[Tuple[int, int, float]]:
    """
    rewards[i][s] = reward of trajectory i at step s (ragged).
    Returns [(traj_idx, step_idx, advantage)] for every existing step with a
    non-zero advantage.
    """
    alive = [i for i, rs in enumerate(rewards) if rs]
    max_len = max((len(rewards[i]) for i in alive), default=0)
    out: List[Tuple[int, int, float]] = []
    for s in range(max_len):
        held = [rewards[i][s] if s < len(rewards[i]) else rewards[i][-1] for i in alive]
        adv = z_score_normalize(held, eps)
        for i, a in zip(alive, adv):
            if s < len(rewards[i]) and abs(a) >= eps:
                out.append((i, s, float(a)))
    return out
