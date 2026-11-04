"""E10 - Recent Voronoi-based planners on the maps and queries of E1 (revision 1, Table 1).

The reimplementations of MVDRM and G2VD (src/baselines/voronoi_roadmaps.py) are run on
exactly the 168 queries of E1 (same maps, same S and T), together with TVE and WDT in
the same execution, so that the four planners of Table 1 are compared on the same
problems and their times are measured in the same session.

    python experiments/e10_voronoi_planners.py [--workers 8] [--quick]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import NAMES, cli, experiment_seeds, metrics, planner_params, run_pool, save_rows, st_columns  # noqa: E402
from e1_comparison import case_queries  # noqa: E402

PLANNERS = ["tve", "teselado", "mvdrm", "g2vd"]


def run_case(scene, level, seed, queries):
    from common.geometry import polyline_length
    from registry import BY_KEY
    from tve import clear_cache
    case, probs, gt = case_queries(scene, level, seed, queries)
    rows = []
    for qi, P in enumerate(probs):
        ref = BY_KEY["visibility"].execute(P)
        ref_len = polyline_length(ref.path) if ref.success else None
        for k in PLANNERS:
            if k == "tve":
                clear_cache()
            res = BY_KEY[k].execute(P, **planner_params(k))
            row = dict(case["info"], query=qi, algo=k, name=NAMES.get(k, k), **st_columns(P))
            row.update(metrics(P, res, ref_len, gt))
            rows.append(row)
    return rows


def jobs(quick=False):
    s = experiment_seeds()["E1"]
    seeds = s["seeds"][:1] if quick else s["seeds"]
    q = 1 if quick else s["queries"]
    return [(sc, lv, k, q) for sc in s["scenes"] for lv in s["levels"] for k in seeds]


def main():
    a = cli(__doc__, "e10.csv")
    df = save_rows(run_pool(run_case, jobs(a.quick), a.workers, "E10"), Path(a.out))
    print(f"E10: {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
