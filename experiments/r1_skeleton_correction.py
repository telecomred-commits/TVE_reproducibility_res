"""Revision 1 - The 12 maps re-executed after the correction of the selection of the skeleton.

For every map of E1-E7 the script rebuilds the configuration space and the two skeletons
of TVE (one generator per obstacle, and contour generators) and lists the maps in which BOTH
report defects: in them the first submission kept the per-obstacle skeleton and the
corrected code keeps the contour skeleton. For each of these maps it reports the defects and
nodes of both skeletons and, from results/raw/original_submission/ (before) and
results/raw/ (after), the length ratio and time of TVE in every query.

    python experiments/r1_skeleton_correction.py [--workers 8]

Output: results/rerun/r1_skeleton_correction_maps.csv and r1_skeleton_correction_queries.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROBOT_D, ROOT, cli, experiment_seeds, run_pool, save_rows  # noqa: E402

E4_SIZES = (200, 300, 400, 600, 800, 1000, 1200)


def check(scene, level, seed, size):
    from common.cspace import configuration_space, robot_radius_px
    from tve.tve import extract_skeleton, free_components, skeleton_defects, voronoi_tessellation
    from _common import build_case
    case = build_case(scene, level, seed, size)
    env = case["env"]
    mpp = env.meters_per_pixel
    occ = configuration_space(env.obstacles, robot_radius_px(ROBOT_D, 0.0, mpp), "disk")
    fc = free_components(occ)
    out = dict(scene=scene, level=level, seed=seed, size=size)
    for mode in ("obstacles", "contours"):
        tess, _ = voronoi_tessellation(occ, mode, 2.0 / mpp)
        g = extract_skeleton(tess, 3.0)
        out[f"defects_{mode}"] = int(skeleton_defects(g, occ, fc))
        out[f"nodes_{mode}"] = len(g.nodes)
        if mode == "obstacles" and out["defects_obstacles"] == 0:
            out["defects_contours"] = out["nodes_contours"] = None
            break
    return [out]


def maps():
    S = experiment_seeds()
    out = {}
    for e, s in S.items():
        if e in ("E8", "E9", "E10"):
            continue
        if e == "E4":
            for sc in s["scenes"]:
                for k in s["seeds"]:
                    for z in E4_SIZES:
                        out.setdefault((sc, "medio", k, z), set()).add(e)
            continue
        for sc in s["scenes"]:
            for lv in s["levels"]:
                for k in s["seeds"]:
                    out.setdefault((sc, lv, k, 400), set()).add(e)
    return out


def main():
    a = cli(__doc__, "r1_skeleton_correction_maps.csv")
    M = maps()
    rows = run_pool(check, sorted(M), a.workers, "R1-skeleton")
    df = pd.DataFrame(rows)
    df["experiments"] = [";".join(sorted(M[(r.scene, r.level, r.seed, r.size)])) for r in df.itertuples()]
    # first submission: the contour skeleton replaced the per-obstacle one only with STRICTLY fewer defects
    df["affected"] = (df.defects_obstacles > 0) & (df.defects_contours.astype(float) >= df.defects_obstacles)
    aff = df[df.affected].sort_values(["experiments", "scene", "level", "seed", "size"])
    print(f"{len(df)} maps scanned, {int((df.defects_obstacles > 0).sum())} with defects in the per-obstacle "
          f"skeleton, {len(aff)} in which both skeletons have defects (re-executed)")
    out = Path(a.out)
    save_rows(aff.to_dict("records"), out)
    # TVE before / after in every query of the affected maps
    raw, old = ROOT / "results" / "raw", ROOT / "results" / "raw" / "original_submission"
    q = []
    spec = {"e1": ("algo", "tve", "relacion_optima", "tiempo_ms"), "e2": ("algo", "tve", "relacion_optima", "t_total_ms"),
            "e4": ("algo", "tve", "ratio", "t_ms"), "e5": ("algo", "tve", "ratio", "t_ms"),
            "e6": ("variant", "TVE (full)", "relacion_optima", "t_ms"), "e7": ("K", 4, "ratio", "t_ms")}
    for e, (col, val, rcol, tcol) in spec.items():
        A, B = pd.read_csv(old / f"{e}.csv"), pd.read_csv(raw / f"{e}.csv")
        for r in aff.itertuples():
            if e.upper() not in r.experiments.split(";"):
                continue
            key = dict(scene=r.scene, seed=r.seed) if e == "e4" else dict(scene=r.scene, level=r.level, seed=r.seed)
            if e == "e4":
                key["size"] = r.size

            def sel(D):
                m = (D[col] == val)
                for k, v in key.items():
                    m &= (D[k] == v)
                return D[m].sort_values("query")
            a_, b_ = sel(A), sel(B)
            for (_, x), (_, y) in zip(a_.iterrows(), b_.iterrows()):
                q.append(dict(experiment=e.upper(), **key, query=int(x["query"]), ratio_before=x.get(rcol),
                              ratio_after=y.get(rcol), time_ms_before=x.get(tcol), time_ms_after=y.get(tcol)))
    save_rows(q, out.with_name(out.stem.replace("_maps", "_queries") + ".csv"))
    print(f"{len(q)} queries of TVE before/after -> {out.parent}")


if __name__ == "__main__":
    main()
