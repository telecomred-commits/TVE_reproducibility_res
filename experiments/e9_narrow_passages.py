"""E9 - Narrow passages (revision 1).

Synthetic maps (scenes/narrow_passages.py) with a passage whose free width in the
configuration space is 1-16 px (0.03-0.48 m at 0.03 m/px), in three geometries
(door in a wall joined to the frame, slit between two separate obstacles, 3-m-long
corridor), 6 layouts per geometry and width, one query per layout. S lies below the
wall and T above it, near the passage, so that the shortest path crosses the passage
and the alternative is a detour through a wide opening. The maps are used directly as
detected maps (no camera).

Recorded per planner: success, validity, length ratio to the visibility graph, and
whether the path crosses the narrow passage.

    python experiments/e9_narrow_passages.py [--workers 8] [--quick]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (ROBOT_D, cli, experiment_seeds, metrics, planner_params, run_pool,  # noqa: E402
                     save_rows, st_columns, GroundTruth)

PLANNERS = ["tve", "teselado", "astar", "theta", "rrt_star", "prm", "mvdrm", "g2vd"]
KINDS = ("door", "slit", "corridor")
GAPS = (1, 2, 3, 4, 6, 8, 12, 16)


def build(kind, gap, seed):
    from common.cspace import configuration_space, robot_radius_px
    from common.environment import Environment
    from common.problem import Problem
    from common import scenarios as sc
    from scenes.narrow_passages import SIDE_M, SIZE, narrow_map
    mpp = SIDE_M / SIZE
    r = robot_radius_px(ROBOT_D, 0.0, mpp)
    gt, box, xd = narrow_map(kind, gap, seed, r)
    env = Environment(gt, mpp, name=f"{kind}-{gap}-{seed}")
    occ = configuration_space(gt, r)
    rng = np.random.default_rng(1000 * gap + seed + {"door": 0, "slit": 7, "corridor": 13}[kind])
    y0, y1 = box[1], box[3]
    for _ in range(400):
        s = np.array([xd + rng.integers(-60, 61), rng.integers(y1 + 25, min(y1 + 110, 390))], float)
        t = np.array([xd + rng.integers(-60, 61), rng.integers(max(y0 - 110, 10), y0 - 25)], float)
        if occ[int(s[1]), int(s[0])] or occ[int(t[1]), int(t[0])]:
            continue
        P = Problem.build(env, s, t, ROBOT_D, occ=occ)
        if P.connected():
            return P, GroundTruth(gt, r), box
    return None, None, box


def crosses(path, box):
    from common.geometry import densify
    if path is None:
        return False
    d = densify(np.asarray(path, float), 0.5)
    x0, y0, x1, y1 = box
    return bool(np.any((d[:, 0] >= x0) & (d[:, 0] <= x1) & (d[:, 1] >= y0) & (d[:, 1] <= y1)))


def run_case(kind, gap, seed):
    from common.geometry import polyline_length
    from registry import BY_KEY
    from tve import clear_cache
    P, gt, box = build(kind, gap, seed)
    if P is None:
        return []
    ref = BY_KEY["visibility"].execute(P)
    ref_len = polyline_length(ref.path) if ref.success else None
    clear_mid = float(P.clearance[(box[1] + box[3]) // 2, (box[0] + box[2]) // 2])
    rows = []
    for k in ["visibility"] + PLANNERS:
        if k == "tve":
            clear_cache()
        res = ref if k == "visibility" else BY_KEY[k].execute(P, **planner_params(k))
        row = dict(kind=kind, gap_px=gap, gap_m=gap * P.mpp, seed=seed, query=0, algo=k, mpp=P.mpp,
                   passage_clearance_px=clear_mid, through_passage=crosses(res.path if res.success else None, box),
                   **st_columns(P))
        row.update(metrics(P, res, ref_len, gt))
        rows.append(row)
    return rows


def jobs(quick=False):
    s = experiment_seeds()["E9"]
    seeds = s["seeds"][:1] if quick else s["seeds"]
    gaps = (1, 4, 16) if quick else GAPS
    return [(kd, g, k) for kd in KINDS for g in gaps for k in seeds]


def main():
    a = cli(__doc__, "e9.csv")
    df = save_rows(run_pool(run_case, jobs(a.quick), min(a.workers, 8), "E9"), Path(a.out))
    print(f"E9: {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
