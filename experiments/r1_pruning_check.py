"""Revision 1 - Effect of the pruning of candidates on the queries of E1.

TVE discards a candidate when the lower bound of its portals is not smaller than the
shortest funnel path found so far, and validates only the four shortest funnel paths. Both
rules compare lengths BEFORE the validation: a funnel path that collides is repaired
afterwards and can become longer, so a discarded candidate could, in principle, have given
a shorter valid path. This script runs TVE on the 168 queries of E1 with the parameters of
E1 (a) as in the paper and (b) without pruning and validating every funnel path, and stores
both lengths and times.

    python experiments/r1_pruning_check.py [--workers 8]

Output: results/rerun/r1_pruning_check.csv (the file used in the paper is in results/raw/).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import TVE_CFG, cli, experiment_seeds, run_pool, save_rows, st_columns, tve_config  # noqa: E402
from e1_comparison import case_queries  # noqa: E402

E1_FILLET_M = TVE_CFG["registry"]["radio_suavizado_m"]          # 1.0 m in E1


def run_case(scene, level, seed, queries):
    from common.geometry import polyline_length
    from tve import TVEPlanner, clear_cache
    _, probs, _ = case_queries(scene, level, seed, queries)
    rows = []
    for qi, P in enumerate(probs):
        out = dict(scene=scene, level=level, seed=seed, query=qi, mpp=P.mpp, **st_columns(P))
        for tag, kw in (("paper", {}), ("no_pruning", dict(prune_bound=False, n_validate=0))):
            clear_cache()
            t0 = time.perf_counter()
            r = TVEPlanner(P.env, tve_config(turn_radius=E1_FILLET_M, **kw)).plan(P.start, P.goal, use_cache=False)
            out[f"t_{tag}_ms"] = 1000 * (time.perf_counter() - t0)
            out[f"ok_{tag}"] = bool(r.success)
            out[f"L_{tag}_px"] = polyline_length(r.path) if r.success else float("nan")
            out[f"L_taut_{tag}_px"] = polyline_length(r.taut) if r.success else float("nan")
            out[f"pruned_{tag}"] = r.stats.get("podadas_cota")
            out[f"candidates_{tag}"] = r.stats.get("candidatas")
        rows.append(out)
    return rows


def jobs(quick=False):
    s = experiment_seeds()["E1"]
    seeds = s["seeds"][:1] if quick else s["seeds"]
    q = 1 if quick else s["queries"]
    return [(sc, lv, k, q) for sc in s["scenes"] for lv in s["levels"] for k in seeds]


def main():
    a = cli(__doc__, "r1_pruning_check.csv")
    df = save_rows(run_pool(run_case, jobs(a.quick), a.workers, "R1-pruning"), Path(a.out))
    print(f"pruning check: {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
