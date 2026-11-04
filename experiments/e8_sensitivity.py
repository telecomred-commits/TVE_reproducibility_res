"""E8 - One-at-a-time sensitivity of the parameters of TVE (revision 1).

7 scenes x 3 levels x 3 layouts (seeds 700-702, maps not used in E1-E7) x 2 queries.
Each parameter of Table 4 is varied while the others keep their values in
configs/tve.json ("planner"); the grid is in configs/tve.json ("sensitivity_E8").
Every run is executed without cache. The reference optimum is the visibility graph.

    python experiments/e8_sensitivity.py [--workers 8] [--quick]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (TVE_CFG, build_case, cli, experiment_seeds, metrics, query_rng, run_pool,  # noqa: E402
                     sample_queries, save_rows, st_columns, tve_config)

GRID = TVE_CFG["sensitivity_E8"]


def settings():
    """(parameter, value) pairs; the default value of each parameter appears once per parameter."""
    out = []
    for name, values in GRID.items():
        for v in values:
            out.append((name, v))
    return out


def case_queries(scene, level, seed, queries):
    case = build_case(scene, level, seed)
    probs, gt = sample_queries(case, queries, query_rng("q8", scene, level, seed))
    return case, probs, gt


def run_case(scene, level, seed, queries):
    from common.geometry import polyline_length
    from common.problem import AlgorithmResult
    from registry import BY_KEY
    from tve import TVEPlanner, clear_cache
    _, probs, gt = case_queries(scene, level, seed, queries)
    rows = []
    for qi, P in enumerate(probs):
        ref = BY_KEY["visibility"].execute(P)
        ref_len = polyline_length(ref.path) if ref.success else None
        for name, value in settings():
            clear_cache()
            t0 = time.perf_counter()
            r = TVEPlanner(P.env, tve_config(**{name: value})).plan(P.start, P.goal, use_cache=False)
            dt = time.perf_counter() - t0
            res = AlgorithmResult("tve", "TVE", r.success, r.path if r.success else None, dt)
            row = dict(scene=scene, level=level, seed=seed, query=qi, parameter=name, value=value,
                       is_default=bool(value == TVE_CFG["planner"][name]), t_ms=1000 * dt, **st_columns(P))
            row.update(metrics(P, res, ref_len, gt))
            rows.append(row)
    return rows


def jobs(quick=False):
    s = experiment_seeds()["E8"]
    seeds = s["seeds"][:1] if quick else s["seeds"]
    q = 1 if quick else s["queries"]
    return [(sc, lv, k, q) for sc in s["scenes"] for lv in s["levels"] for k in seeds]


def main():
    a = cli(__doc__, "e8.csv")
    df = save_rows(run_pool(run_case, jobs(a.quick), min(a.workers, 8), "E8"), Path(a.out))
    print(f"E8: {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
