"""Revision 1: re-executes the 12 map jobs affected by the correction of the selection of the skeleton
and splices them into the results of the first submission.

The correction (src/tve/tve.py::TVEPlanner.prepare) only changes the maps in which both skeletons of TVE
(one generator per obstacle, contour generators) report defects; they are listed in
results/raw/r1_skeleton_correction_maps.csv (experiments/r1_skeleton_correction.py finds them). Each
affected job is run again with one worker and replaces, in its own position, ALL the rows of that job in
results/raw/original_submission/e*.csv, so that paired comparisons come from the same execution. E4 is
re-executed too, but its results did not change and the published file keeps its original rows.

    python tools/rerun_affected.py

Output: results/rerun/spliced/e*.csv. Compare them with the published files with
    python verify.py --rerun results/rerun/spliced
(only the timing columns of the re-executed jobs differ, since they come from a new execution).
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from _common import run_pool  # noqa: E402

LOG = json.loads((ROOT / "results" / "raw" / "revision1_reruns.json").read_text(encoding="utf-8"))
MODULES = {"e1": "e1_comparison", "e2": "e2_tve_vs_wdt", "e4": "e4_scalability", "e5": "e5_replanning",
           "e6": "e6_ablation", "e7": "e7_homotopy"}
GROUP = {"e4": ["scene", "seed", "size"]}


def main():
    out_dir = ROOT / "results" / "rerun" / "spliced"
    out_dir.mkdir(parents=True, exist_ok=True)
    for exp, info in LOG.items():
        jobs = [tuple(j) for j in info["jobs"]]
        old = pd.read_csv(ROOT / "results" / "raw" / "original_submission" / f"{exp}.csv")
        if exp == "e4":                                   # unchanged: the published file keeps the original rows
            old.to_csv(out_dir / f"{exp}.csv", index=False)
            continue
        m = importlib.import_module(MODULES[exp])
        new = pd.DataFrame(run_pool(m.run_case, jobs, 1, exp.upper()))[old.columns]
        g = GROUP.get(exp, ["scene", "level", "seed"])
        k = g + [c for c in ("query", "algo", "variant", "K") if c in old.columns]
        pos = old.reset_index().merge(new, on=k, suffixes=("", "_new"), how="inner")
        out = old.copy()
        at = pos["index"].to_numpy()
        for c in old.columns:
            if c in k:
                continue
            col = out[c].astype(object).to_numpy(copy=True)
            col[at] = pos[f"{c}_new"].astype(object).to_numpy()
            out[c] = pd.Series(col, index=out.index).infer_objects()
        out.to_csv(out_dir / f"{exp}.csv", index=False)
        print(exp, len(jobs), "jobs,", len(at), "rows replaced", flush=True)


if __name__ == "__main__":
    main()
