from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from player.policy import Theta
from player.rollout import simulate_packed


def _resolve_workers(workers: int) -> int:
    if workers <= 0:
        return max(1, os.cpu_count() or 4)
    return workers


def _rollout_seed(base_seed: int, adam_iter: int, tag: int, rollout_index: int) -> int:
    x = base_seed ^ (adam_iter * 1_009_663) ^ (tag * 100_003) ^ (rollout_index * 917_521)
    return x & 0x7FFFFFFF


def _run_simulate_scores(
    tasks: list[tuple[Theta, int, tuple[int, int], int]],
    workers: int,
) -> list[float]:
    w = _resolve_workers(workers)
    if not tasks:
        return []
    if w <= 1:
        return [simulate_packed(t) for t in tasks]
    chunksize = max(1, len(tasks) // (w * 8))
    with ProcessPoolExecutor(max_workers=w) as pool:
        return list(pool.map(simulate_packed, tasks, chunksize=chunksize))


def mean_objective(
    theta: Theta,
    *,
    n_rollouts: int,
    max_steps: int,
    field_size: tuple[int, int],
    base_seed: int,
    adam_iter: int,
    tag: int,
    workers: int,
) -> float:
    """Среднее J(θ) по ``n_rollouts`` независимым партиям."""
    tasks: list[tuple[Theta, int, tuple[int, int], int]] = [
        (theta, max_steps, field_size, _rollout_seed(base_seed, adam_iter, tag, k)) for k in range(n_rollouts)
    ]
    scores = _run_simulate_scores(tasks, workers)
    return sum(scores) / len(scores)


def _perturb(theta: Theta, axis: int, delta: float) -> Theta:
    lst = list(theta)
    lst[axis] += delta
    return (lst[0], lst[1], lst[2], lst[3])


def center_value_and_fd_grad(
    theta: Theta,
    *,
    eps: float,
    rollouts_per_arm: int,
    rollouts_center: int,
    max_steps: int,
    field_size: tuple[int, int],
    base_seed: int,
    adam_iter: int,
    workers: int,
) -> tuple[float, tuple[float, float, float, float]]:
    """
    Одна пачка симуляций на шаг Adam:
    среднее J(θ) по ``rollouts_center`` и градиент центральными разностями
    (каждое плечо — ``rollouts_per_arm`` партий).
    """
    if eps <= 0.0:
        msg = "eps для конечных разностей должен быть > 0"
        raise ValueError(msg)

    tasks: list[tuple[Theta, int, tuple[int, int], int]] = []
    # центр: теги 0 .. rollouts_center-1
    for k in range(rollouts_center):
        tasks.append((theta, max_steps, field_size, _rollout_seed(base_seed, adam_iter, 0, k)))

    rpa = rollouts_per_arm
    # для каждой координаты: +ε, −ε; теги 100+2*i, 100+2*i+1
    for i in range(4):
        th_p = _perturb(theta, i, eps)
        th_m = _perturb(theta, i, -eps)
        tag_p = 100 + 2 * i
        tag_m = 100 + 2 * i + 1
        for k in range(rpa):
            tasks.append((th_p, max_steps, field_size, _rollout_seed(base_seed, adam_iter, tag_p, k)))
        for k in range(rpa):
            tasks.append((th_m, max_steps, field_size, _rollout_seed(base_seed, adam_iter, tag_m, k)))

    scores = _run_simulate_scores(tasks, workers)
    off = 0
    j_center = sum(scores[off : off + rollouts_center]) / rollouts_center
    off += rollouts_center

    grad: list[float] = []
    for _i in range(4):
        j_plus = sum(scores[off : off + rpa]) / rpa
        off += rpa
        j_minus = sum(scores[off : off + rpa]) / rpa
        off += rpa
        grad.append((j_plus - j_minus) / (2.0 * eps))
    return j_center, (grad[0], grad[1], grad[2], grad[3])


@dataclass
class AdamState:
    """Первые и вторые моменты градиента (максимизация J)."""

    m: list[float]
    v: list[float]
    t: int

    @classmethod
    def zeros(cls, dim: int = 4) -> AdamState:
        return cls(m=[0.0] * dim, v=[0.0] * dim, t=0)


def adam_max_step(
    theta: Theta,
    grad: tuple[float, float, float, float],
    state: AdamState,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> tuple[Theta, AdamState]:
    """Один шаг Adam по направлению градиента J (максимизация)."""
    state.t += 1
    t = state.t
    out: list[float] = []
    for i in range(4):
        g = grad[i]
        state.m[i] = beta1 * state.m[i] + (1.0 - beta1) * g
        state.v[i] = beta2 * state.v[i] + (1.0 - beta2) * g * g
        m_hat = state.m[i] / (1.0 - beta1**t)
        v_hat = state.v[i] / (1.0 - beta2**t)
        step = lr * m_hat / (v_hat**0.5 + eps)
        out.append(theta[i] + step)
    return (out[0], out[1], out[2], out[3]), state
