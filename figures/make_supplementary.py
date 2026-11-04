"""Supplementary Figure S1: the 84 detected maps of E1 with the paths of TVE, WDT and the optimum.

One page per scene type (3 difficulty levels x 4 layouts). Each panel shows the
rectified camera image with the detected obstacles, the configuration-space band and,
for the two queries of the map, the paths of TVE (solid), WDT (dashed) and the
visibility graph (dotted, reference optimum L*), with the length ratio of TVE and WDT.
The planners are re-run with the parameters of E1 (configs/), so the paths are those
of results/raw/e1.csv.

    python figures/make_supplementary.py [--workers 8]

Output: figures/output/figS1_e1_maps.pdf
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import _common as X  # noqa: E402

CM = 1 / 2.54


def paths_of_map(scene, level, seed):
    from common.geometry import polyline_length
    from registry import BY_KEY
    from tve import clear_cache
    from e1_comparison import case_queries
    case, probs, _ = case_queries(scene, level, seed, X.experiment_seeds()["E1"]["queries"])
    out = []
    for P in probs:
        ref = BY_KEY["visibility"].execute(P)
        clear_cache()
        tve = BY_KEY["tve"].execute(P, **X.planner_params("tve"))
        wdt = BY_KEY["teselado"].execute(P, **X.planner_params("teselado"))
        L0 = polyline_length(ref.path) if ref.success else np.nan
        q = dict(S=P.start, T=P.goal, ref=ref.path if ref.success else None)
        for k, r in (("tve", tve), ("wdt", wdt)):
            q[k] = r.path if r.success else None
            q[k + "_ratio"] = polyline_length(r.path) / L0 if r.success else np.nan
        out.append(q)
    occ = probs[0].occ if probs else None
    return dict(scene=scene, level=level, seed=seed, image=case["perception"].image, obstacles=case["env"].obstacles,
                occ=occ, mpp=case["env"].meters_per_pixel, queries=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=X.default_workers())
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from scenes.generators import LABELS
    s = X.experiment_seeds()["E1"]
    jobs = [(sc, lv, k) for sc in s["scenes"] for lv in s["levels"] for k in s["seeds"]]
    with ProcessPoolExecutor(a.workers, initializer=X._init_worker) as ex:
        maps = list(ex.map(paths_of_map, *zip(*jobs)))
    by = {(m["scene"], m["level"], m["seed"]): m for m in maps}
    # consistency check: the ratios printed in the figure must be those of results/raw/e1.csv
    import pandas as pd
    raw = pd.read_csv(ROOT / "results" / "raw" / "e1.csv")
    raw = raw[raw.algo.isin(["tve", "teselado"])].set_index(["scene", "level", "seed", "query", "algo"])
    rows = []
    for m in maps:
        for qi, q in enumerate(m["queries"]):
            for k, key in (("tve", "tve"), ("wdt", "teselado")):
                ref = float(raw.loc[(m["scene"], m["level"], m["seed"], qi, key), "relacion_optima"])
                rows.append(dict(scene=m["scene"], level=m["level"], seed=m["seed"], query=qi, planner=key,
                                 ratio_figure=q[k + "_ratio"], ratio_raw=ref))
    chk = pd.DataFrame(rows)
    chk["abs_diff"] = (chk.ratio_figure - chk.ratio_raw).abs()
    out = ROOT / "figures" / "output" / "figS1_e1_maps.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    chk.to_csv(out.with_name("figS1_ratios_check.csv"), index=False, float_format="%.6g")
    print(f"ratios in the figure vs results/raw/e1.csv: {len(chk)} values, max |difference| = {chk.abs_diff.max():.2e}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
    styles = {"ref": ((0, (1, 1)), "#4a3aa7", 1.0), "wdt": ("--", "#eb6834", 1.0), "tve": ("-", "#2a78d6", 1.3)}
    with PdfPages(out) as pdf:
        # cover page
        fig = plt.figure(figsize=(18.47 * CM, 20.5 * CM))
        fig.text(0.06, 0.93, "Supplementary Figure S1", fontsize=12, weight="bold")
        fig.text(0.06, 0.89, "The 84 detected maps of experiment E1 and the paths of TVE, WDT and the visibility graph",
                 fontsize=9)
        body = (
            "Manuscript robotics-4632217, revision 1 (O. Penagos, I. Ladino, C. R. Suárez).\n\n"
            "One page per type of scene (office, warehouse, apartment, workshop, retail store, outdoor plaza and\n"
            "hospital ward). Rows: low, medium and high difficulty of the image; columns: layouts 0 to 3.\n\n"
            "Each panel shows the rectified camera image (400 x 400 px), the obstacles detected by the image\n"
            "processing (red), the band added by the dilation by the robot radius (light red), and, for the two\n"
            "queries of the map (1 and 2: circle = S, star = T), the paths of TVE (solid blue), WDT (dashed orange)\n"
            "and the visibility graph (dotted purple), whose length is the reference L*. Below each panel: the\n"
            "length ratio L/L* of TVE and WDT for each query. The bar measures 2 m; the floor sides are 12, 16,\n"
            "10, 12, 14, 18 and 14 m, in the order above.\n\n"
            "The planners are executed again with the parameters of E1 (configs/), and the ratios printed in the\n"
            f"panels were checked against results/raw/e1.csv: {len(chk)} values, largest difference "
            f"{chk.abs_diff.max():.1e}.\n\n"
            "Produced by figures/make_supplementary.py of the reproducibility package.")
        fig.text(0.06, 0.84, body, fontsize=8, va="top", linespacing=1.5)
        pdf.savefig(fig)
        plt.close(fig)
        for sc in s["scenes"]:
            fig, axs = plt.subplots(3, 4, figsize=(18.47 * CM, 20.5 * CM))
            for i, lv in enumerate(s["levels"]):
                for j, sd in enumerate(s["seeds"]):
                    ax = axs[i, j]
                    m = by[(sc, lv, sd)]
                    ax.imshow(m["image"])
                    if m["occ"] is not None:
                        band = m["occ"] & ~m["obstacles"]
                        ax.imshow(np.ma.masked_where(~band, band), cmap=ListedColormap(["#e34948"]), alpha=0.18)
                    ax.imshow(np.ma.masked_where(~m["obstacles"], m["obstacles"]),
                              cmap=ListedColormap(["#e34948"]), alpha=0.5)
                    txt = []
                    for qi, q in enumerate(m["queries"]):
                        for k in ("ref", "wdt", "tve"):
                            if q[k] is not None:
                                ls, c, lw = styles[k]
                                ax.plot(q[k][:, 0], q[k][:, 1], linestyle=ls, color=c, lw=lw)
                        for p, mk in ((q["S"], "o"), (q["T"], "*")):
                            ax.scatter(*p, s=14 if mk == "o" else 26, c="#ffffff", edgecolors="#0b0b0b",
                                       linewidths=0.6, marker=mk, zorder=9)
                        ax.annotate(f"{qi + 1}", q["S"], xytext=(3, 3), textcoords="offset points", fontsize=7)
                        txt.append(f"q{qi + 1}: TVE {q['tve_ratio']:.3f}, WDT {q['wdt_ratio']:.3f}")
                    # 2-m scale bar
                    h, w = m["obstacles"].shape
                    L = 2.0 / m["mpp"]
                    ax.plot([w * 0.04, w * 0.04 + L], [h * 0.965] * 2, color="#0b0b0b", lw=2.0, solid_capstyle="butt")
                    ax.text(w * 0.04 + L / 2, h * 0.94, "2 m", ha="center", va="bottom", fontsize=7,
                            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
                    ax.set_title(f"{X.LEVEL_EN[lv]} difficulty, layout {sd}", fontsize=7, pad=2)
                    ax.set_xlabel("\n".join(txt), fontsize=7, labelpad=2)
                    ax.set_xticks([])
                    ax.set_yticks([])
            fig.suptitle(f"Figure S1 ({LABELS[sc]}): detected maps of E1 and paths "
                         f"(L/L* of TVE and WDT for each query)", fontsize=8)
            fig.legend([Line2D([], [], color=styles[k][1], ls=styles[k][0], lw=styles[k][2]) for k in ("tve", "wdt", "ref")],
                       ["TVE (proposed)", "WDT (original method)", "Visibility graph (optimum L*)"],
                       loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
            fig.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=2.2, w_pad=0.4)
            pdf.savefig(fig)
            plt.close(fig)
            print("  page", sc, flush=True)
    print("written", out)


if __name__ == "__main__":
    main()
