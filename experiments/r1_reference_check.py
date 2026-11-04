"""Revision 1 - Sensitivity of the reference optimum L* (visibility graph) on the queries of E1.

The reference of the paper, L*, is the visibility-graph path among the convex vertices of
the contours of C_obs simplified with a tolerance of 1 px (Douglas-Peucker) and moved 2 px
towards the free space (or 3 / 4.5 px when those points are occupied). Every such path is
free (exact test), so L* is an UPPER BOUND of the Euclidean shortest path in the pixel
configuration space; the simplification and the offset of the vertices can make it a little
longer than the true optimum, and the offset closes passages narrower than about 4 px.

This script recomputes, for the 168 queries of E1, a tighter reference L*_t (tolerance
0.5 px, offset 0.75 px, i.e. just outside the corner of the occupied pixel) and stores both,
together with the lengths of the planners taken from results/raw/e1.csv.

    python experiments/r1_reference_check.py [--workers 8]

Output: results/rerun/r1_reference_check.csv (the file used in the paper is in results/raw/).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import cli, experiment_seeds, run_pool, save_rows, st_columns  # noqa: E402
from e1_comparison import case_queries  # noqa: E402

TIGHT = dict(tolerancia_poligono=0.5, desplazamiento_px=0.75)


def run_case(scene, level, seed, queries):
    from common.geometry import polyline_length
    from registry import BY_KEY
    _, probs, _ = case_queries(scene, level, seed, queries)
    rows = []
    for qi, P in enumerate(probs):
        out = dict(scene=scene, level=level, seed=seed, query=qi, mpp=P.mpp, **st_columns(P))
        for tag, kw in (("default", {}), ("tight", TIGHT)):
            t0 = time.perf_counter()
            r = BY_KEY["visibility"].execute(P, **kw)
            out[f"L_ref_{tag}_px"] = polyline_length(r.path) if r.success else float("nan")
            out[f"n_vertices_{tag}"] = r.stats.get("vertices")
            out[f"t_ref_{tag}_ms"] = 1000 * (time.perf_counter() - t0)
        rows.append(out)
    return rows


def jobs(quick=False):
    s = experiment_seeds()["E1"]
    seeds = s["seeds"][:1] if quick else s["seeds"]
    q = 1 if quick else s["queries"]
    return [(sc, lv, k, q) for sc in s["scenes"] for lv in s["levels"] for k in seeds]


def main():
    a = cli(__doc__, "r1_reference_check.csv")
    df = save_rows(run_pool(run_case, jobs(a.quick), a.workers, "R1-reference"), Path(a.out))
    print(f"reference check: {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
