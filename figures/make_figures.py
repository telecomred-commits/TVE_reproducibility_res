"""FIGURES step: rebuilds the figures of the paper (revision 1).

    python figures/make_figures.py                    # all figures from results/raw
    python figures/make_figures.py --raw results/rerun
    python figures/make_figures.py --only quantitative   (or: qualitative)

Figures 1 and 2 (block and flow diagrams) are TikZ drawings included with the manuscript
sources. Quantitative figures are computed from the raw CSV files. Qualitative figures
(pipeline, gallery, paths) re-run the planners on fixed scenes and queries; they use the
planner defaults of the common interface, which for these illustrative figures means a
fillet radius of 1.0 m for TVE and a turning radius of 1.0 m for Hybrid A* (paths figure).
Every figure is drawn at its final printed width (13.86 cm = text width, or 18.47 cm =
full page width of the MDPI template) with the same 7-pt font, so that no figure is
rescaled in the manuscript; maps carry a 2-m scale bar. Vector PDF; raster PNG at 300 dpi.
The supplementary figure with the 84 maps of E1 is produced by figures/make_supplementary.py.

Outputs: figures/output/*.pdf and *.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import _common as X  # noqa: E402  (sets paths and threads)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = ROOT / "figures" / "output"
CM = 1 / 2.54
W1, W2 = 13.86 * CM, 18.47 * CM          # text width and full width of the template
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COL = {"tve": "#2a78d6", "teselado": "#eb6834", "astar": "#1baf7a", "theta": "#eda100", "rrt_star": "#e87ba4",
       "hybrid_astar": "#008300", "visibility": "#4a3aa7", "aco": "#e34948", "mvdrm": "#8c564b",
       "g2vd": "#17becf", "prm": "#7f7f7f"}
LS = {"tve": "-", "teselado": "--", "astar": ":", "theta": "-.", "rrt_star": (0, (5, 1.5)),
      "hybrid_astar": (0, (3, 1, 1, 1)), "visibility": (0, (1, 1)), "aco": (0, (6, 2, 1, 2)),
      "mvdrm": (0, (2, 1)), "g2vd": (0, (4, 1, 1, 1, 1, 1)), "prm": (0, (1, 2))}
NAMES, LEVELS, LEVEL_EN = X.NAMES, X.LEVELS, X.LEVEL_EN
FS = 7.0                                  # small text (annotations, legends)
KEY = ["scene", "level", "seed", "query"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 7, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "lines.linewidth": 1.4,
    "pdf.fonttype": 42,
})


def col(k):
    return COL.get(k, "#b9b7af")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("  figure", name)


def labels():
    from scenes.generators import LABELS
    return LABELS


def _valid(d):
    return d["exito"].astype(bool) & d["sin_colision"].fillna(False).astype(bool)


# --------------------------------------------------------------------------- qualitative (Figures 2-4)
def _style_map(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def _scalebar(ax, mpp, shape, length_m=2.0):
    """2-m scale bar in the lower-left corner of a map panel (pixel coordinates)."""
    h, w = shape[:2]
    L = length_m / mpp
    x0, y0 = 0.05 * w, 0.955 * h
    ax.plot([x0, x0 + L], [y0, y0], color=INK, lw=2.2, solid_capstyle="butt", zorder=11)
    ax.text(x0 + L / 2, y0 - 0.015 * h, f"{length_m:g} m", ha="center", va="bottom", fontsize=FS, color=INK,
            zorder=11, bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.75))


def _st(ax, P, s=26):
    ax.scatter(*P.start, s=s, c="#ffffff", edgecolors=INK, linewidths=0.8, zorder=9, marker="o")
    ax.scatter(*P.goal, s=s * 1.6, c="#ffffff", edgecolors=INK, linewidths=0.8, zorder=9, marker="*")
    ax.annotate("S", P.start, xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK, zorder=10)
    ax.annotate("T", P.goal, xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK, zorder=10)


def fig_pipeline():
    """Figure 2: stages of TVE on the office scene (medium level, layout 7)."""
    from matplotlib.colors import ListedColormap
    from tve import TVEPlanner, clear_cache
    case = X.build_case("office", "medio", 7)
    P = X.sample_queries(case, 1, np.random.default_rng(3), min_sep=0.6)[0][0]
    clear_cache()
    tp = TVEPlanner(P.env, X.tve_config())
    r = tp.plan(P.start, P.goal, use_cache=False)
    p = tp.prep
    res = case["perception"]
    fig, axs = plt.subplots(2, 3, figsize=(W2, 11.6 * CM))
    a = axs.ravel()
    a[0].imshow(case["photo"])
    if case["corners"] is not None:
        c = np.vstack([case["corners"], case["corners"][:1]])
        a[0].plot(c[:, 0], c[:, 1], "-", color="#eda100", lw=1.2)
    a[0].set_title("(a) camera image and floor quadrilateral")
    a[1].imshow(res.overlay(alpha=0.45, color=(227, 73, 72)))
    a[1].set_title("(b) rectified image, detected obstacles")
    lab = p.tess.labels.astype(float)
    pal = ListedColormap(np.random.default_rng(1).uniform(0.55, 0.95, (max(2, int(lab.max()) + 1), 3)))
    a[2].imshow(np.ma.masked_where(p.occ, lab), cmap=pal, interpolation="nearest")
    a[2].imshow(np.ma.masked_where(~p.occ, p.occ), cmap=ListedColormap(["#3a3a38"]), interpolation="nearest")
    a[2].imshow(np.ma.masked_where(~P.env.obstacles, P.env.obstacles), cmap=ListedColormap(["#0b0b0b"]),
                interpolation="nearest")
    a[2].set_title("(c) C-space and Euclidean Voronoi tiles")
    a[3].imshow(np.where(p.occ, 0.0, 1.0), cmap="gray", vmin=-0.3, vmax=1)
    segs, vals = [], []
    for e in p.graph.edges.values():
        if len(e.pts) < 2:
            continue
        segs.append(np.stack([e.pts[:-1], e.pts[1:]], 1))
        ci = np.clip(np.rint(e.pts[:-1]).astype(int), 0, np.array(p.occ.shape[::-1]) - 1)
        vals.append(p.checker.dt[ci[:, 1], ci[:, 0]] * P.mpp)
    lc = LineCollection(np.vstack(segs), array=np.concatenate(vals), cmap="viridis", linewidths=1.3)
    a[3].add_collection(lc)
    nodes = np.array([xy for n, xy in p.graph.nodes.items() if p.graph.degree(n) >= 3])
    if len(nodes):
        a[3].scatter(nodes[:, 0], nodes[:, 1], s=7, c="#e34948", zorder=5, linewidths=0)
    cb = fig.colorbar(lc, ax=a[3], fraction=0.046, pad=0.02)
    cb.set_label("clearance weight [m]", fontsize=FS)
    cb.ax.tick_params(labelsize=6)
    a[3].set_title("(d) weighted skeleton and nodes")
    a[4].imshow(np.where(p.occ, 0.0, 1.0), cmap="gray", vmin=-0.3, vmax=1)
    for x in r.routes:
        a[4].plot(x["polyline"][:, 0], x["polyline"][:, 1], "-", color="#86b6ef", lw=0.9)
    por = [(u, v) for u, v in r.portals[::2] if np.hypot(*(np.asarray(u) - v)) > 1e-6]
    a[4].add_collection(LineCollection(por, colors="#eda100", linewidths=0.5))
    a[4].plot(r.taut[:, 0], r.taut[:, 1], "-o", color=COL["tve"], lw=1.3, ms=2.2)
    a[4].set_title("(e) K candidate routes, portals, funnel")
    a[5].imshow(res.image)
    a[5].imshow(np.ma.masked_where(~p.occ, p.occ), cmap=ListedColormap(["#e34948"]), alpha=0.25)
    a[5].plot(r.path[:, 0], r.path[:, 1], "-", color=COL["tve"], lw=1.8)
    a[5].set_title(f"(f) arc-smoothed path, R ≥ {X.R_MIN:.1f} m")
    for ax in a:
        _style_map(ax)
    for ax in a[1:]:
        _st(ax, P)
        _scalebar(ax, P.mpp, p.occ.shape)
    fig.tight_layout(w_pad=0.6, h_pad=0.8)
    save(fig, "fig_pipeline")


def fig_gallery():
    """Gallery: one scene of each type (layouts 11-17) with the paths of TVE and WDT.

    Four columns (camera image above, detected map below), so that every map is printed
    about 4.4 cm wide."""
    from baselines.wdt.planner import Planner
    from tve import TVEPlanner, clear_cache
    L = labels()
    levels = ["facil", "medio", "dificil", "medio", "dificil", "facil", "medio"]
    fig = plt.figure(figsize=(W2, 16.6 * CM))
    gs = fig.add_gridspec(4, 4, height_ratios=[3.3, 4.4, 3.3, 4.4], hspace=0.16, wspace=0.05)
    for j, (s, lv) in enumerate(zip(L, levels)):
        r0, c = (0, j) if j < 4 else (2, j - 4)
        ax_p, ax_m = fig.add_subplot(gs[r0, c]), fig.add_subplot(gs[r0 + 1, c])
        case = X.build_case(s, lv, 11 + j)
        probs, _ = X.sample_queries(case, 1, np.random.default_rng(j), min_sep=0.6)
        ax_p.imshow(case["photo"])
        ax_p.set_title(f"{L[s]}\n({LEVEL_EN[lv]} difficulty)", fontsize=7, pad=2)
        ax_m.imshow(case["perception"].image)
        ax_m.imshow(np.ma.masked_where(~case["env"].obstacles, case["env"].obstacles),
                    cmap=matplotlib.colors.ListedColormap(["#e34948"]), alpha=0.45)
        if probs:
            P = probs[0]
            clear_cache()
            r = TVEPlanner(P.env, X.tve_config()).plan(P.start, P.goal)
            w = Planner(P.env, X.wdt_config()).plan(P.start, P.goal)
            if w.success:
                ax_m.plot(w.final_path[:, 0], w.final_path[:, 1], LS["teselado"], color=COL["teselado"], lw=1.3)
            if r.success:
                ax_m.plot(r.path[:, 0], r.path[:, 1], "-", color=COL["tve"], lw=1.6)
            _st(ax_m, P, s=18)
        _scalebar(ax_m, case["env"].meters_per_pixel, case["env"].obstacles.shape)
        for ax in (ax_p, ax_m):
            _style_map(ax)
    fig.legend([Line2D([], [], color=COL["tve"], lw=1.6), Line2D([], [], color=COL["teselado"], ls="--", lw=1.3),
                Line2D([], [], color="#e34948", lw=5, alpha=0.45)],
               [NAMES["tve"], "WDT (original method)", "detected obstacles"], loc="center",
               bbox_to_anchor=(0.88, 0.25), frameon=False, ncol=1)
    save(fig, "fig_gallery")


def fig_paths():
    """Figure 4: paths of seven planners on three detected maps (layouts 21-23)."""
    from registry import BY_KEY
    from tve import clear_cache
    L = labels()
    keys = ["tve", "teselado", "astar", "theta", "rrt_star", "hybrid_astar", "aco"]
    cases = [("warehouse", "medio", 21), ("home", "dificil", 22), ("plaza", "medio", 23)]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 6.4 * CM))
    for ax, (s, lv, sd) in zip(axs, cases):
        case = X.build_case(s, lv, sd)
        P = X.sample_queries(case, 1, np.random.default_rng(sd), min_sep=0.65)[0][0]
        ax.imshow(np.where(P.occ, 0.82, 1.0), cmap="gray", vmin=0, vmax=1)
        ax.imshow(np.ma.masked_where(~P.env.obstacles, P.env.obstacles),
                  cmap=matplotlib.colors.ListedColormap(["#52514e"]))
        for k in keys[::-1]:
            clear_cache()
            res = BY_KEY[k].execute(P)                 # interface defaults, as in the paper
            if res.success:
                ax.plot(res.path[:, 0], res.path[:, 1], linestyle=LS[k], color=col(k),
                        lw=1.6 if k == "tve" else 1.0, zorder=6 if k == "tve" else 4)
        _st(ax, P)
        _scalebar(ax, P.mpp, P.occ.shape)
        _style_map(ax)
        ax.set_title(f"{L[s]} ({LEVEL_EN[lv]} difficulty)", fontsize=7)
    handles = [Line2D([], [], color=col(k), ls=LS[k], lw=1.6 if k == "tve" else 1.1) for k in keys]
    fig.legend(handles, [NAMES[k] for k in keys], loc="lower center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, -0.06), handlelength=2.6)
    fig.tight_layout(w_pad=0.4)
    save(fig, "fig_paths")


# --------------------------------------------------------------------------- quantitative (Figures 5-11)
def figs_e1(raw):
    d = pd.read_csv(raw / "e1.csv")
    d["valid"] = _valid(d)
    ok = d[d.valid]
    go = ok.groupby("algo")
    order = list(pd.DataFrame({"L": go["relacion_optima"].median(), "t": go["tiempo_ms"].median()})
                 .sort_values(["L", "t"]).index)
    # Figure 5: quality-time plane
    fig, ax = plt.subplots(figsize=(W1, 7.0 * CM))
    for k in order:
        s = ok[ok.algo == k]
        x, y = s.tiempo_ms.median(), s.relacion_optima.median()
        xl, xh = s.tiempo_ms.quantile([0.25, 0.75])
        yl, yh = s.relacion_optima.quantile([0.25, 0.75])
        ax.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]], fmt="o", ms=5.5 if k == "tve" else 4,
                    color=col(k), ecolor=col(k) if k in COL else "#c9c7bf", elinewidth=0.7, capsize=0, zorder=5,
                    mec="#ffffff", mew=0.6)
        off = {"astar": (4, -10), "rrt_star": (-8, -11), "tve": (-64, 4), "hybrid_astar": (6, 4), "theta": (6, 5),
               "topological": (-30, -12), "visibility": (-40, 6), "teselado": (5, 3), "dijkstra": (5, 3),
               "prm": (5, 3), "slam": (5, 3), "aco": (5, 3)}.get(k, (4, 3))
        ax.annotate(NAMES[k], (x, y), xytext=off, textcoords="offset points", fontsize=FS,
                    color=INK if k == "tve" else INK2, fontweight="bold" if k == "tve" else "normal")
    ax.set_xscale("log")
    ax.set_ylim(0.99, min(1.6, ax.get_ylim()[1]))
    ax.set_xlabel("median computation time [ms] (log); bars: interquartile range")
    ax.set_ylabel("median length / optimal")
    save(fig, "fig_pareto")
    # Figure 6: distributions
    fig, axs = plt.subplots(1, 2, figsize=(W2, 7.4 * CM))
    for ax, cn, lab, logx in ((axs[0], "tiempo_ms", "computation time [ms] (log)", True),
                              (axs[1], "relacion_optima", "path length / optimal length", False)):
        vals = [ok[ok.algo == k][cn].dropna().to_numpy() for k in order]
        if cn == "relacion_optima":
            vals = [np.clip(v, None, 2.0) for v in vals]
        bp = ax.boxplot(vals, orientation="horizontal", widths=0.62, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.0), whiskerprops=dict(color=MUTED, lw=0.7),
                        capprops=dict(color=MUTED, lw=0.7), boxprops=dict(lw=0.6, color=MUTED))
        for patch, k in zip(bp["boxes"], order):
            patch.set_facecolor(col(k) if k in ("tve", "teselado") else "#dcdad3")
            patch.set_edgecolor(col(k) if k in ("tve", "teselado") else MUTED)
        ax.set_yticks(range(1, len(order) + 1))
        ax.set_yticklabels([NAMES[k] for k in order])
        ax.invert_yaxis()
        if logx:
            ax.set_xscale("log")
        else:
            ax.set_xlim(0.98, 1.6)
            ax.axvline(1.0, color=MUTED, lw=0.7, ls="--")
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
    axs[1].set_yticklabels([])
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_boxplots")
    # Figure 8: difficulty levels
    fig, axs = plt.subplots(1, 2, figsize=(W2, 5.6 * CM))
    sel = ["tve", "teselado", "astar", "theta", "rrt_star", "hybrid_astar", "aco", "visibility"]
    x, bw = np.arange(3), 0.105
    for i, k in enumerate(sel):
        succ = [100 * d[(d.algo == k) & (d.level == L_)].valid.mean() for L_ in LEVELS]
        ack = [100 * ok[(ok.algo == k) & (ok.level == L_)].factible_ackermann.mean() for L_ in LEVELS]
        off = (i - (len(sel) - 1) / 2) * bw
        axs[0].bar(x + off, succ, bw * 0.9, color=col(k), label=NAMES[k])
        axs[1].bar(x + off, ack, bw * 0.9, color=col(k))
    for ax, t in zip(axs, ("valid path found [%]", "Ackermann-feasible paths [%]")):
        ax.set_xticks(x)
        ax.set_xticklabels([LEVEL_EN[L_] for L_ in LEVELS])
        ax.set_xlabel("image-processing difficulty")
        ax.set_ylabel(t)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylim(50, 101)
    axs[0].legend(ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=FS)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_levels")
    # Figure 9: minimum turning radius
    fig, ax = plt.subplots(figsize=(W1, 5.8 * CM))
    for k in ["tve", "teselado", "hybrid_astar", "rrt_star", "theta", "astar", "visibility"]:
        v = np.sort(np.clip(ok[ok.algo == k].radio_giro_min_m.replace(np.inf, 50).to_numpy(), 0.01, 50))
        if len(v):
            ax.step(v, 1 - np.arange(len(v)) / len(v), where="post", color=col(k), ls=LS[k],
                    lw=1.2 if k == "tve" else 1.0, label=NAMES[k])
    ax.axvline(X.R_MIN, color=INK2, lw=0.8)
    ax.axvline(0.97 * X.R_MIN, color=INK2, lw=0.8, ls="--")          # threshold of Ackermann feasibility
    ax.annotate(f"$R_{{\\min}}$ = {X.R_MIN} m", (X.R_MIN, 1.0), xytext=(-4, 6), textcoords="offset points",
                fontsize=FS, color=INK2, ha="right", va="bottom")
    ax.set_ylim(-0.02, 1.12)
    ax.set_xscale("log")
    ax.set_xlabel("minimum turning radius along the path [m] (log)")
    ax.set_ylabel("fraction of paths with radius ≥ x")
    ax.legend(frameon=False, ncol=2, fontsize=FS)
    save(fig, "fig_curvature")


def fig_e2(raw):
    """Figure 7: stage times, paired speed-up and length ratios of TVE and WDT."""
    d = pd.read_csv(raw / "e2.csv")
    d["valid"] = _valid(d)
    t = d.pivot_table(index=KEY, columns="algo", values="t_total_ms").dropna()
    L = d[d.valid].pivot_table(index=KEY, columns="algo", values="relacion_optima").dropna()
    sp = t["teselado"] / t["tve"]
    stages = {"teselado": ["t_dilatacion_ms", "t_teselado_ms", "t_esqueleto_ms", "t_conexion_ST_ms", "t_rutas_ms",
                           "t_reduccion_ms", "t_suavizado_ms"],
              "tve": ["t_dilatacion_ms", "t_teselado_ms", "t_esqueleto_ms", "t_conexion_ST_ms", "t_rutas_ms",
                      "t_embudo_ms", "t_suavizado_ms"]}
    names = ["dilation", "tessellation", "skeleton", "S/T connection", "route search", "reduction / funnel", "smoothing"]
    cols = ["#8c8a83", "#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#eb6834", "#4a3aa7"]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 6.4 * CM), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    for i, a in enumerate(("teselado", "tve")):
        v = d[d.algo == a][stages[a]].median().to_numpy()
        left = 0
        for j, (val, c) in enumerate(zip(v, cols)):
            val = 0 if not np.isfinite(val) else val
            axs[0].barh(i, val, left=left, color=c, height=0.55, edgecolor="#ffffff", linewidth=0.8,
                        label=names[j] if i == 0 else None)
            left += val
    axs[0].set_yticks([0, 1])
    axs[0].set_yticklabels(["WDT", "TVE"])
    axs[0].set_xlabel("median time per stage [ms]")
    axs[0].legend(frameon=False, fontsize=FS, ncol=2, loc="lower left", bbox_to_anchor=(-0.02, 1.0),
                  handlelength=1.2, columnspacing=0.8)
    axs[0].grid(axis="y", visible=False)
    axs[1].hist(np.log2(sp), bins=30, color=COL["tve"], edgecolor="#ffffff", linewidth=0.5)
    axs[1].axvline(0, color=INK2, lw=0.8)
    axs[1].set_xlabel("speed-up WDT / TVE (log$_2$)")
    axs[1].set_ylabel("queries")
    ticks = [-1, 0, 1, 2, 3, 4]
    axs[1].set_xticks(ticks)
    axs[1].set_xticklabels([f"{2 ** v:g}×" for v in ticks])
    lim = (0.98, max(1.35, float(np.nanquantile(L.to_numpy(), 0.99))))
    axs[2].scatter(L["teselado"], L["tve"], s=5, color=COL["tve"], alpha=0.55, linewidths=0)
    axs[2].plot(lim, lim, color=INK2, lw=0.8)
    axs[2].set_xlim(lim)
    axs[2].set_ylim(lim)
    axs[2].set_xlabel("WDT length / optimal")
    axs[2].set_ylabel("TVE length / optimal")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_wdt")


def fig_e3(raw):
    """Figure 10: cumulative time of consecutive queries."""
    d = pd.read_csv(raw / "e3.csv")
    fig, ax = plt.subplots(figsize=(W1, 5.8 * CM))
    for k in ["tve", "teselado", "astar", "theta", "prm", "visibility"]:
        s = d[d.algo == k].sort_values("query")
        s = s.assign(cum=s.groupby(["scene", "seed"])["t_ms"].cumsum())
        m = s.groupby("query")["cum"].median() / 1000
        ax.plot(m.index + 1, m.to_numpy(), color=col(k), ls=LS.get(k, "-"), lw=1.5 if k == "tve" else 1.0,
                label=NAMES[k])
    ax.set_yscale("log")
    ax.set_xlabel("number of queries on the same map")
    ax.set_ylabel("cumulative time [s] (log)")
    ax.legend(frameon=False, ncol=3, fontsize=FS)
    save(fig, "fig_multiquery")


def fig_e4(raw):
    """Figure 11: computation time against the map size."""
    d = pd.read_csv(raw / "e4.csv")
    d = d[d.ok.astype(bool)]
    fig, ax = plt.subplots(figsize=(W1, 6.0 * CM))
    extra = {"dijkstra": ("#6f6d67", (0, (4, 1.5))), "prm": ("#b9b7af", "-")}
    for k in ["tve", "teselado", "astar", "dijkstra", "theta", "prm", "visibility"]:
        s = d[d.algo == k].groupby("pixels")["t_ms"].median()
        if len(s) < 2:
            continue
        b = np.polyfit(np.log(s.index.to_numpy(float)), np.log(s.to_numpy()), 1)[0]
        c_, ls_ = extra.get(k, (col(k), LS.get(k, "-")))
        ax.plot(s.index, s.to_numpy(), marker="o", ms=3, color=c_, ls=ls_, lw=1.5 if k == "tve" else 1.0,
                label=f"{NAMES[k]} (k = {b:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("map size N [pixels] (log)")
    ax.set_ylabel("median computation time [ms] (log)")
    ax.legend(frameon=False, fontsize=FS, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_scaling")


def fig_e9(raw):
    """Narrow passages (E9): share of queries whose valid path crosses the passage, by free width."""
    f = raw / "e9.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d["ok"] = _valid(d) & d.through_passage.astype(bool)
    keys = ["tve", "teselado", "mvdrm", "g2vd", "theta", "astar", "rrt_star", "prm", "visibility"]
    titles = {"door": "door in a wall joined to the frame", "slit": "slit between two obstacles",
              "corridor": "corridor 3 m long"}
    fig, axs = plt.subplots(1, 3, figsize=(W2, 5.4 * CM), sharey=True)
    for ax, kind in zip(axs, ("door", "slit", "corridor")):
        s = d[d.kind == kind]
        for k in keys[::-1]:
            m = 100 * s[s.algo == k].groupby("gap_px").ok.mean()
            if len(m):
                ax.plot(m.index, m.to_numpy(), marker="o", ms=2.5, color=col(k), ls=LS.get(k, "-"),
                        lw=1.6 if k == "tve" else 1.0, zorder=6 if k == "tve" else 4)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16])
        mpp = float(s.mpp.iloc[0])
        ax.set_xticklabels([f"{g}\n({g * mpp:.2f})" for g in (1, 2, 4, 8, 16)])
        ax.set_xlabel("free width of the passage [px] ([m])")
        ax.set_title(titles[kind])
    axs[0].set_ylabel("paths through the passage [%]")
    axs[0].set_ylim(-4, 104)
    handles = [Line2D([], [], color=col(k), ls=LS.get(k, "-"), marker="o", ms=2.5, lw=1.6 if k == "tve" else 1.0)
               for k in keys]
    fig.legend(handles, [NAMES.get(k, k).replace("G2VD", "G$^2$VD") for k in keys], loc="lower center", ncol=9,
               frameon=False, bbox_to_anchor=(0.5, -0.07), handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_narrow")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
    ap.add_argument("--only", choices=("quantitative", "qualitative"))
    a = ap.parse_args()
    raw = Path(a.raw)
    if a.only != "qualitative":
        figs_e1(raw)
        fig_e2(raw)
        fig_e3(raw)
        fig_e4(raw)
        fig_e9(raw)
    if a.only != "quantitative":
        fig_pipeline()
        fig_gallery()
        fig_paths()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
