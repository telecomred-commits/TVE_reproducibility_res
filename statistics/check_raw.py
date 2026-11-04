"""Integrity check of the raw results (revision 1).

For every experiment it checks that the raw file contains exactly the runs of the design
in seeds/experiment_seeds.csv and configs/, that the key columns have no missing values,
and, for E10, that the queries are exactly those of E1 (same maps, S and T). It also writes
the SHA-256 checksum of every file in results/raw/.

    python statistics/check_raw.py

Outputs: results/raw_files_check.txt, results/checksums_sha256.txt
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import _common as X  # noqa: E402

RAW = ROOT / "results" / "raw"


def design():
    S = X.experiment_seeds()
    n_maps = {e: len(s["scenes"]) * len(s["levels"]) * len(s["seeds"]) for e, s in S.items()}
    tve = json.loads((ROOT / "configs" / "tve.json").read_text(encoding="utf-8"))
    n_settings = sum(len(v) for v in tve["sensitivity_E8"].values())
    return S, n_maps, n_settings


def main():
    S, n_maps, n_settings = design()
    lines, ok_all = [], True

    def check(name, cond, detail):
        nonlocal ok_all
        ok_all &= bool(cond)
        lines.append(f"[{'OK' if cond else 'FAIL'}] {name}: {detail}")

    e1 = pd.read_csv(RAW / "e1.csv")
    k1 = ["scene", "level", "seed", "query"]
    check("E1 runs", len(e1) == n_maps["E1"] * S["E1"]["queries"] * len(X.ALGOS),
          f"{len(e1)} rows = {n_maps['E1']} maps x {S['E1']['queries']} queries x {len(X.ALGOS)} planners")
    # E8
    e8 = pd.read_csv(RAW / "e8.csv")
    exp8 = n_maps["E8"] * S["E8"]["queries"] * n_settings
    check("E8 runs", len(e8) == exp8, f"{len(e8)} rows = {n_maps['E8']} maps x {S['E8']['queries']} queries x "
          f"{n_settings} settings")
    per_q = e8.groupby(k1).size()
    check("E8 every query has every setting", (per_q == n_settings).all(), f"{len(per_q)} queries")
    check("E8 one default row per parameter and query",
          (e8[e8.is_default.astype(bool)].groupby(k1 + ["parameter"]).size() == 1).all(), "is_default")
    check("E8 seeds not used in E1-E7", set(e8.seed) == set(S["E8"]["seeds"]), f"seeds {sorted(set(e8.seed))}")
    # E9
    e9 = pd.read_csv(RAW / "e9.csv")
    gaps = sorted(e9.gap_px.unique())
    n9 = len(e9.kind.unique()) * len(gaps) * len(S["E9"]["seeds"])
    check("E9 runs", len(e9) == n9 * 9, f"{len(e9)} rows = 3 geometries x {len(gaps)} widths x "
          f"{len(S['E9']['seeds'])} layouts x 9 planners (visibility + 8)")
    check("E9 every planner succeeded or failed explicitly", e9.exito.notna().all(), "column exito")
    # E10
    e10 = pd.read_csv(RAW / "e10.csv")
    check("E10 runs", len(e10) == n_maps["E10"] * S["E10"]["queries"] * 4,
          f"{len(e10)} rows = {n_maps['E10']} maps x {S['E10']['queries']} queries x 4 planners")
    st = ["sx", "sy", "tx", "ty"]
    # S and T of E1 as exported by scenes/generate.py (the raw file of E1 predates the sx..ty columns)
    a = pd.read_csv(ROOT / "seeds" / "queries_E1.csv").set_index(k1)[st]
    b = e10.drop_duplicates(k1).set_index(k1)[st]
    same = a.reindex(b.index).sub(b).abs().to_numpy().max()
    check("E10 uses the queries of E1", same < 1e-9 and len(b) == len(a), f"max |difference of S, T| = {same:g} px")
    t1 = e1[e1.algo == "tve"].set_index(k1)
    t10 = e10[e10.algo == "tve"].set_index(k1)
    dL = float((t1.longitud_m.reindex(t10.index) - t10.longitud_m).abs().max())
    check("E10 TVE paths equal to those of E1", dL < 1e-6, f"max |difference of length| = {dL:.1e} m")
    # missing values in the key metric columns of the valid runs
    for name, df in (("E1", e1), ("E8", e8), ("E9", e9), ("E10", e10)):
        v = df[df.exito.astype(bool) & df.sin_colision.fillna(False).astype(bool)]
        miss = int(v[["longitud_m", "tiempo_ms", "holgura_min_m"]].isna().sum().sum())
        no_ref = v[v.relacion_optima.isna()].groupby(["kind", "gap_px", "seed"] if name == "E9" else k1).ngroups
        check(f"{name} valid runs have all metrics", miss == 0,
              f"{len(v)} valid runs, {miss} missing values; queries without reference L* (the visibility graph "
              f"found no path): {no_ref}")
    # revision-1 files
    log = json.loads((RAW / "revision1_reruns.json").read_text(encoding="utf-8"))
    n_jobs = sum(len(v["jobs"]) for v in log.values())
    m = pd.read_csv(RAW / "r1_skeleton_correction_maps.csv")
    check("revision log lists the 12 maps", n_jobs == 12 and len(m) == 12, f"{n_jobs} jobs in revision1_reruns.json, "
          f"{len(m)} maps in r1_skeleton_correction_maps.csv")
    for f in sorted((RAW / "original_submission").glob("e*.csv")):
        o, n = pd.read_csv(f), pd.read_csv(RAW / f.name)
        check(f"original_submission/{f.name}", len(o) == len(n), f"{len(o)} rows (same runs as the corrected file)")
    # checksums
    sums = []
    for f in sorted(RAW.rglob("*")):
        if f.is_file():
            sums.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(ROOT).as_posix()}")
    (ROOT / "results" / "checksums_sha256.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    lines.append(f"\nSHA-256 of {len(sums)} files written to results/checksums_sha256.txt")
    lines.insert(0, "Integrity check of results/raw (python statistics/check_raw.py)\n")
    (ROOT / "results" / "raw_files_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
