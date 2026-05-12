from __future__ import annotations

import json

from player.policy import Theta

type ResultRow = tuple[Theta, float]

# Доля итераций под «широкий ландшафт» и под «крупные шаги»; остальное — локальная доводка
_EXPLORE_END = 0.34
_ADVANCE_END = 0.68


def search_schedule(
    iter_1based: int,
    total_iterations: int,
    *,
    n_candidates: int = 3,
    plateau_break: bool = False,
) -> dict[str, object]:
    """
    Удешевляет вызовы LLM: сначала разведка противоположными точками, затем крупные сдвиги, потом мелкие.
    iter_1based: 1 .. total_iterations
    """
    n = max(1, int(n_candidates))
    if total_iterations <= 0:
        total_iterations = 1
    if iter_1based < 1:
        iter_1based = 1
    if iter_1based > total_iterations:
        iter_1based = total_iterations
    if total_iterations == 1:
        t = 1.0
    else:
        t = (iter_1based - 1) / (total_iterations - 1)
    if plateau_break:
        phase = "plateau_break"
        instruction = (
            f"Mean score stagnated. Return exactly {n} vectors pairwise FAR in R^4; "
            "do NOT repeat near-identical rows; avoid clustering on current best; "
            "explore different quadrants of weight space."
        )
    elif t < _EXPLORE_END:
        phase = "explore"
        instruction = (
            f"{n} candidates must be MAXIMALLY different from EACH OTHER (large pairwise distance in R^4); "
            "cover the search space, not near copies of listed points. Explore the landscape."
        )
    elif t < _ADVANCE_END:
        phase = "advance"
        instruction = (
            f"{n} candidates: LARGE moves biased by best trials, steer away from worst; "
            "may cluster only if all clearly improve score."
        )
    else:
        phase = "refine"
        instruction = (
            f"{n} candidates: SMALL local steps around the best theta (fine-tune); shrink exploration."
        )
    return {
        "iteration": iter_1based,
        "total": total_iterations,
        "phase": phase,
        "n_candidates": n,
        "plateau_break": plateau_break,
        "instruction": instruction,
    }


SYSTEM_PROMPT = """You are an optimization engine for four continuous policy weights (Snake policy).
Each step you read schedule, best/worst, mean_score, and optimization_trajectory: past evaluated
(theta, mean J) pairs sorted by score ascending (low to high), with i = iteration index when observed.
Use that trajectory like in-context demonstrations: notice clusters among higher scores and steer
candidates accordingly, without contradicting the trajectory evidence.
You form ONE sentence hypothesis_note (goal + hypothesis + link to weights), then candidates.
Follow schedule for exploration vs refinement. Never duplicate candidate rows.
Return ONLY valid JSON matching the schema. No prose outside JSON."""


def build_optimization_trajectory(
    entries: list[tuple[int, tuple[float, float, float, float], float]],
    *,
    max_send: int = 72,
) -> list[dict[str, object]]:
    """
    Траектория для метазапроса: все накопленные точки, сортировка по score по возрастанию;
    при переполнении — равномерная прорежка по рангу после сортировки.
    entries: (iteration_1based, theta, mean_J)
    """
    if not entries:
        return []
    by_score = sorted(entries, key=lambda e: e[2])
    if len(by_score) <= max_send:
        thin = by_score
    else:
        n = len(by_score)
        idxs = sorted(
            {min(n - 1, max(0, round(j * (n - 1) / max(max_send - 1, 1)))) for j in range(max_send)}
        )
        thin = [by_score[j] for j in idxs]
    return [
        {"i": i, "theta": [round(x, 3) for x in t], "score": round(float(s), 3)}
        for i, t, s in thin
    ]


def build_llm_user_payload(
    results: list[ResultRow],
    *,
    top_k: int = 5,
    worst_k: int = 5,
    iteration: int | None = None,
    total_iterations: int | None = None,
    n_llm_candidates: int = 3,
    plateau_break: bool = False,
    trajectory_entries: list[tuple[int, tuple[float, float, float, float], float]] | None = None,
    trajectory_max_send: int = 72,
) -> dict[str, object]:
    if not results:
        out: dict[str, object] = {
            "top_k": top_k,
            "worst_k": worst_k,
            "best": [],
            "worst": [],
            "mean_score": 0.0,
        }
        if iteration is not None and total_iterations is not None:
            out["schedule"] = search_schedule(
                iteration,
                total_iterations,
                n_candidates=n_llm_candidates,
                plateau_break=plateau_break,
            )
        if trajectory_entries:
            out["optimization_trajectory"] = build_optimization_trajectory(
                trajectory_entries,
                max_send=trajectory_max_send,
            )
        return out
    by_desc = sorted(results, key=lambda x: x[1], reverse=True)
    best = by_desc[:top_k]
    # Худшие по J среди рангов вне top_k — без дубля одного и того же запуска в best и worst
    outside_top = by_desc[top_k:]
    worst = outside_top[-worst_k:] if worst_k and outside_top else []
    worst.sort(key=lambda x: x[1])
    mean_score = sum(s for _, s in results) / len(results)
    payload: dict[str, object] = {
        "top_k": top_k,
        "worst_k": worst_k,
        "best": [{"theta": [round(x, 3) for x in t], "score": round(s, 3)} for t, s in best],
        "worst": [{"theta": [round(x, 3) for x in t], "score": round(s, 3)} for t, s in worst],
        "mean_score": round(mean_score, 3),
    }
    if iteration is not None and total_iterations is not None:
        payload["schedule"] = search_schedule(
            iteration,
            total_iterations,
            n_candidates=n_llm_candidates,
            plateau_break=plateau_break,
        )
    if trajectory_entries:
        payload["optimization_trajectory"] = build_optimization_trajectory(
            trajectory_entries,
            max_send=trajectory_max_send,
        )
    return payload


def user_message_content(payload: dict[str, object], *, n_candidates: int = 3) -> str:
    n = max(1, int(n_candidates))
    extra = {
        "task": "maximize apples collected in N steps with death penalty",
        "output_format": (
            f"JSON with hypothesis_note (one sentence: goal + hypothesis + link to weights) "
            f"and exactly {n} distinct candidate vectors (4 floats each); "
            "use optimization_trajectory (score ascending) as optimization trace context"
        ),
    }
    merged = {**payload, **extra}
    return json.dumps(merged, ensure_ascii=False, separators=(",", ":"))
