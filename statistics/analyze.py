"""ANALYZE step: raw results -> tables of the paper (CSV + LaTeX) and every number in the text.

    python statistics/analyze.py                         # uses results/raw (the runs of the paper)
    python statistics/analyze.py --raw results/rerun     # uses a new execution

Outputs (in results/tables/ unless --out is given):
    table05_e1_results.csv          Table 5   E1, all planners
    table06_ranks_wilcoxon.csv      Table 6   Friedman mean ranks, Wilcoxon + Holm, effect sizes
    table07_e1_by_scene.csv         Table 7   E1 by scene
    table08_e2_by_scene.csv         Table 8   E2 paired TVE vs WDT by scene (bootstrap CI)
    table09_perception_levels.csv   Table 9   image processing and safety by difficulty level
    table10_replanning.csv          Table 10  E5
    table11_ablation.csv            Table 11  E6
    table12_homotopy.csv            Table 12  E7
    e2_stage_times.csv, e3_multiquery.csv, e4_scaling.csv   data of Figures 7, 10, 11
    statistics_summary.json         all tests (chi2, z, p, r, CI) and the numbers quoted in the text
    latex/*.tex, latex/numbers.tex  the same tables as LaTeX bodies and \\newcommand macros
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
from nonparametric import bootstrap_median, friedman, holm, nemenyi_cd, wilcoxon  # noqa: E402

EXP = json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))
TVE_CFG = json.loads((ROOT / "configs" / "tve.json").read_text(encoding="utf-8"))
NAMES = EXP["planner_names_in_paper"]
LEVELS = EXP["levels"]
LEVEL_EN = EXP["level_names_in_paper"]
KEY = ["scene", "level", "seed", "query"]


def scene_labels():
    from scenes.generators import LABELS
    return LABELS


def fmt_p(p):
    if p < 1e-4:
        e = int(math.floor(math.log10(p)))
        return f"\\ensuremath{{{p / 10 ** e:.1f}\\times 10^{{{e}}}}}"
    return f"{p:.4f}"


def prel(p):
    return "<10^{-100}" if p < 1e-100 else "=" + fmt_p(p)


def cite_name(k):
    if k == "g2vd":
        return "G$^2$VD"
    return NAMES[k] + ("~\\cite{ladino2020}" if k == "teselado" else "")


class Writer:
    def __init__(self, out: Path):
        self.out = out
        (out / "latex").mkdir(parents=True, exist_ok=True)

    def csv(self, name, df, index=True):
        df.to_csv(self.out / f"{name}.csv", index=index, float_format="%.6g")
        print("  table", name)

    def tex(self, name, lines):
        (self.out / "latex" / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def valid(d):
    return d["exito"].astype(bool) & d["sin_colision"].fillna(False).astype(bool)


# --------------------------------------------------------------------------- E1
def e1(raw, W, S):
    d = pd.read_csv(raw / "e1.csv")
    d["valid"] = valid(d)
    d["gt_safe"] = d["gt_safe"].fillna(False).astype(bool)
    ok = d[d.valid]
    g, go = d.groupby("algo"), ok.groupby("algo")
    T = pd.DataFrame({
        "valid_pct": 100 * g["valid"].mean(), "L_ratio_median": go["relacion_optima"].median(),
        "L_ratio_p90": go["relacion_optima"].quantile(0.9), "time_ms_median": go["tiempo_ms"].median(),
        "time_ms_p90": go["tiempo_ms"].quantile(0.9), "min_clearance_m": go["holgura_min_m"].median(),
        "sharp_turns": go["giros_bruscos"].median(), "diff_drive_time_s": go["tiempo_diferencial_s"].median(),
        "ackermann_pct": 100 * go["factible_ackermann"].mean(), "gt_safe_pct": 100 * go["gt_safe"].mean(),
    }).sort_values(["L_ratio_median", "time_ms_median"])
    T.insert(0, "planner", [NAMES[k] for k in T.index])
    W.csv("table05_e1_results", T)
    W.tex("t_main", [("\\textbf{TVE (proposed)}" if k == "tve" else cite_name(k)) +
                     f" & {r.valid_pct:.1f} & {r.L_ratio_median:.3f} & {r.L_ratio_p90:.3f} & {r.time_ms_median:.0f} & "
                     f"{r.time_ms_p90:.0f} & {r.min_clearance_m:.2f} & {r.sharp_turns:.0f} & {r.diff_drive_time_s:.1f} & "
                     f"{r.ackermann_pct:.1f} & {r.gt_safe_pct:.1f} \\\\" for k, r in T.iterrows()])
    # by scene
    rows, lines = [], []
    for s, lab in scene_labels().items():
        ds = ok[ok.scene == s].groupby("algo")
        L, t = ds["relacion_optima"].median(), ds["tiempo_ms"].median()
        succ = 100 * d[d.scene == s].groupby("algo")["valid"].mean()
        best = L.drop(["tve", "visibility"]).idxmin()
        rows.append(dict(scene=lab, tve_valid_pct=succ["tve"], tve_L_ratio=L["tve"], tve_time_ms=t["tve"],
                         wdt_valid_pct=succ["teselado"], wdt_L_ratio=L["teselado"], wdt_time_ms=t["teselado"],
                         best_other=NAMES[best], best_other_L_ratio=L[best], best_other_time_ms=t[best]))
        lines.append(f"{lab} & {succ['tve']:.0f} & {L['tve']:.3f} & {t['tve']:.0f} & {succ['teselado']:.0f} & "
                     f"{L['teselado']:.3f} & {t['teselado']:.0f} & {NAMES[best]} ({L[best]:.3f}, {t[best]:.0f} ms) \\\\")
    W.csv("table07_e1_by_scene", pd.DataFrame(rows), index=False)
    W.tex("t_scene", lines)
    # Friedman (planners with >= 95 % valid paths, queries solved by all of them)
    good = [k for k in T.index if T.loc[k, "valid_pct"] >= 95]
    pl = ok.pivot_table(index=KEY, columns="algo", values="relacion_optima")[good].dropna()
    pt = ok.pivot_table(index=KEY, columns="algo", values="tiempo_ms")[good].dropna()
    chi2, p, Rm = friedman(pl.to_numpy())
    chi2t, ptt, Rmt = friedman(pt.to_numpy())
    rk = pd.DataFrame({"rank_L": Rm, "rank_time": Rmt}, index=good).sort_values("rank_L")
    wl = []
    for a in good:
        if a == "tve":
            continue
        pair = ok.pivot_table(index=KEY, columns="algo", values="relacion_optima")[["tve", a]].dropna()
        pairt = ok.pivot_table(index=KEY, columns="algo", values="tiempo_ms")[["tve", a]].dropna()
        _, zL, pL, rL = wilcoxon(pair["tve"], pair[a])
        _, zT, pT, rT = wilcoxon(pairt["tve"], pairt[a])
        wl.append(dict(algo=a, n=len(pair), z_L=zL, p_L=pL, r_L=rL, tve_shorter_pct=100 * float((pair["tve"] < pair[a] - 1e-9).mean()),
                       z_time=zT, p_time=pT, r_time=rT, tve_faster_pct=100 * float((pairt["tve"] < pairt[a]).mean())))
    Wt = pd.DataFrame(wl)
    Wt["p_L_holm"] = holm(Wt["p_L"])
    Wt["p_time_holm"] = holm(Wt["p_time"])
    Wt = Wt.set_index("algo").loc[[a for a in rk.index if a != "tve"]]
    Wt.insert(0, "rank_time", rk.loc[Wt.index, "rank_time"])
    Wt.insert(0, "rank_L", rk.loc[Wt.index, "rank_L"])
    Wt.insert(0, "planner", [NAMES[k] for k in Wt.index])
    W.csv("table06_ranks_wilcoxon", Wt)
    W.tex("t_wilcoxon", [f"{cite_name(a)} & {r.rank_L:.2f} & {r.rank_time:.2f} & {r.tve_shorter_pct:.0f} & {r.r_L:+.2f} & "
                         f"{fmt_p(r.p_L_holm)} & {r.tve_faster_pct:.0f} & {r.r_time:+.2f} & {fmt_p(r.p_time_holm)} \\\\"
                         for a, r in Wt.iterrows()])
    # difficulty levels
    lv = []
    for L_ in LEVELS:
        dl = d[d.level == L_]
        per = dl.drop_duplicates(["scene", "seed"])
        okl = dl[dl.valid]
        lv.append(dict(level=LEVEL_EN[L_], iou=per.iou.mean(), recall=per.recall.mean(),
                       objects_detected_pct=100 * (per.objects_found / per.objects_total.clip(lower=1)).mean(),
                       perception_time_ms=per.t_perception_ms.median(),
                       tve_valid_pct=100 * dl[dl.algo == "tve"].valid.mean(),
                       tve_gt_safe_pct=100 * okl[okl.algo == "tve"].gt_safe.mean(),
                       all_gt_safe_pct=100 * okl.gt_safe.mean()))
    LV = pd.DataFrame(lv)
    W.csv("table09_perception_levels", LV, index=False)
    W.tex("t_levels", [f"{r.level.capitalize()} & {r.iou:.3f} & {r.recall:.3f} & {r.objects_detected_pct:.1f} & "
                       f"{r.perception_time_ms:.0f} & {r.tve_valid_pct:.1f} & {r.tve_gt_safe_pct:.1f} & "
                       f"{r.all_gt_safe_pct:.1f} \\\\" for r in LV.itertuples()])
    S["E1"] = dict(n_maps=int(d.drop_duplicates(["scene", "level", "seed"]).shape[0]), n_queries=int(d.groupby(KEY).ngroups),
                   n_runs=int(len(d)), friedman_L=dict(chi2=chi2, p=p, n=len(pl), k=len(good)),
                   friedman_time=dict(chi2=chi2t, p=ptt, n=len(pt), k=len(good)),
                   nemenyi_cd=nemenyi_cd(len(good), len(pl)))
    return T, rk, LV


# --------------------------------------------------------------------------- E2
def e2(raw, W, S):
    d = pd.read_csv(raw / "e2.csv")
    d["valid"] = valid(d)
    t = d.pivot_table(index=KEY, columns="algo", values="t_total_ms").dropna()
    L = d[d.valid].pivot_table(index=KEY, columns="algo", values="relacion_optima").dropna()
    sp = t["teselado"] / t["tve"]
    med, lo, hi = bootstrap_median(sp)
    _, zt, pt, rt = wilcoxon(t["tve"], t["teselado"])
    _, zl, pl, rl = wilcoxon(L["tve"], L["teselado"])
    stages = ["t_dilatacion_ms", "t_teselado_ms", "t_esqueleto_ms", "t_conexion_ST_ms", "t_rutas_ms",
              "t_reduccion_ms", "t_embudo_ms", "t_suavizado_ms"]
    ST = pd.DataFrame({a: d[d.algo == a][stages].median() for a in ("teselado", "tve")}).T
    ST.index = ["WDT", "TVE"]
    ST.columns = ["dilation", "tessellation", "skeleton", "S_T_connection", "route_search", "midpoint_reduction",
                  "funnel_and_validation", "smoothing"]
    W.csv("e2_stage_times_ms", ST)
    rows, lines = [], []
    for s, lab in scene_labels().items():
        ts, Ls = t.xs(s, level="scene"), L.xs(s, level="scene")
        m, l_, h_ = bootstrap_median(ts["teselado"] / ts["tve"])
        _, _, p_, _ = wilcoxon(ts["tve"], ts["teselado"])
        rows.append(dict(scene=lab, n=len(ts), wdt_ms=ts["teselado"].median(), tve_ms=ts["tve"].median(),
                         speedup=m, ci_low=l_, ci_high=h_, p=p_, wdt_L_ratio=Ls["teselado"].median(),
                         tve_L_ratio=Ls["tve"].median()))
        lines.append(f"{lab} & {len(ts)} & {ts['teselado'].median():.1f} & {ts['tve'].median():.1f} & "
                     f"{m:.2f} [{l_:.2f}, {h_:.2f}] & {fmt_p(p_)} & {Ls['teselado'].median():.3f} & "
                     f"{Ls['tve'].median():.3f} \\\\")
    W.csv("table08_e2_by_scene", pd.DataFrame(rows), index=False)
    W.tex("t_wdt_scene", lines)
    S["E2"] = dict(n=len(t), speedup_median=med, speedup_ci=[lo, hi], tve_faster_pct=100 * float((sp > 1).mean()),
                   time_wilcoxon=dict(z=zt, p=pt, r=rt), length_wilcoxon=dict(z=zl, p=pl, r=rl),
                   tve_shorter_pct=100 * float((L["tve"] < L["teselado"]).mean()),
                   time_ms_median=dict(tve=t["tve"].median(), wdt=t["teselado"].median()),
                   L_ratio_median=dict(tve=L["tve"].median(), wdt=L["teselado"].median()),
                   L_ratio_p90=dict(tve=L["tve"].quantile(0.9), wdt=L["teselado"].quantile(0.9)),
                   ackermann_pct=dict(tve=100 * d[(d.algo == "tve") & d.valid].factible_ackermann.mean(),
                                      wdt=100 * d[(d.algo == "teselado") & d.valid].factible_ackermann.mean()),
                   stage_times_ms=ST.to_dict(orient="index"))


# --------------------------------------------------------------------------- E3 - E7
def e3(raw, W, S):
    d = pd.read_csv(raw / "e3.csv")
    out, summ = [], {}
    for k in ["tve", "teselado", "astar", "theta", "prm", "visibility"]:
        s = d[d.algo == k].sort_values("query")
        s = s.assign(cum=s.groupby(["scene", "seed"])["t_ms"].cumsum())
        m = s.groupby("query")["cum"].median() / 1000
        out.append(pd.DataFrame({"planner": NAMES[k], "queries": m.index + 1, "cumulative_time_s_median": m.to_numpy()}))
        summ[k] = dict(first_ms=s[s["query"] == 0].t_ms.median(), next_ms=s[s["query"] > 0].t_ms.median(),
                       total_s=float(m.iloc[-1]))
    W.csv("e3_multiquery", pd.concat(out), index=False)
    S["E3"] = summ


def e4(raw, W, S):
    d = pd.read_csv(raw / "e4.csv")
    d = d[d.ok.astype(bool)]
    med = d.groupby(["algo", "pixels"])[["t_ms", "t_tess_ms"]].median().reset_index()
    med.insert(0, "planner", med.algo.map(NAMES))
    W.csv("e4_scaling", med, index=False)
    slopes = {}
    for k, s in med.groupby("algo"):
        if len(s) >= 2:
            slopes[k] = float(np.polyfit(np.log(s.pixels.to_numpy(float)), np.log(s.t_ms.to_numpy()), 1)[0])
    for k in ("tve", "teselado"):
        s = med[med.algo == k].dropna(subset=["t_tess_ms"])
        if len(s) >= 2:
            slopes[f"tess_{k}"] = float(np.polyfit(np.log(s.pixels.to_numpy(float)), np.log(s.t_tess_ms.to_numpy()), 1)[0])
    big = d.pixels.max()
    tb = d[d.pixels == big].groupby("algo")
    S["E4"] = dict(slopes=slopes, largest_side_px=int(round(math.sqrt(big))), time_ms_largest=tb["t_ms"].median().to_dict(),
                   tess_ms_largest=tb["t_tess_ms"].median().dropna().to_dict())


def e5(raw, W, S):
    d = pd.read_csv(raw / "e5.csv")
    keys = ["tve", "teselado", "astar", "theta", "rrt", "prm", "hybrid_astar", "visibility"]
    rows, lines = [], []
    for k in keys:
        s = d[d.algo == k]
        o = s[s.ok.astype(bool)]
        rows.append(dict(planner=NAMES[k], success_pct=100 * s.ok.mean(), time_ms_median=o.t_ms.median(),
                         time_ms_p90=o.t_ms.quantile(0.9), L_ratio=o.ratio.median(), L_over_L0=o.detour.median()))
        lines.append(f"{cite_name(k)} & {100 * s.ok.mean():.1f} & {o.t_ms.median():.1f} & {o.t_ms.quantile(0.9):.1f} & "
                     f"{o.ratio.median():.3f} & {o.detour.median():.3f} \\\\")
    R = pd.DataFrame(rows)
    W.csv("table10_replanning", R, index=False)
    W.tex("t_dynamic", lines)
    S["E5"] = dict(n=int(d.groupby(KEY).ngroups), table=R.set_index("planner").to_dict(orient="index"))


def e6(raw, W, S):
    d = pd.read_csv(raw / "e6.csv")
    d["valid"] = valid(d)
    base = d[d.variant == "TVE (full)"].set_index(KEY)
    rows, lines = [], []
    for v in TVE_CFG["ablation_E6"]:
        s = d[d.variant == v]
        o = s[s.valid]
        rel = (s.set_index(KEY).t_ms / base.t_ms).median()
        rows.append(dict(variant=v, valid_pct=100 * s.valid.mean(), time_ms=o.t_ms.median(), relative_time=rel,
                         L_ratio=o.relacion_optima.median(), L_ratio_p90=o.relacion_optima.quantile(0.9),
                         ackermann_pct=100 * o.factible_ackermann.mean()))
        lines.append(f"{v} & {100 * s.valid.mean():.1f} & {o.t_ms.median():.1f} & {rel:.2f} & "
                     f"{o.relacion_optima.median():.3f} & {o.relacion_optima.quantile(0.9):.3f} & "
                     f"{100 * o.factible_ackermann.mean():.1f} \\\\")
    W.csv("table11_ablation", pd.DataFrame(rows), index=False)
    W.tex("t_ablation", lines)


def e7(raw, W, S):
    d = pd.read_csv(raw / "e7.csv")
    d = d[d.ok.astype(bool)]
    rows, lines = [], []
    for K, s in d.groupby("K"):
        r = dict(K=int(K), candidates=s.n_candidates.mean(), classes=s.n_classes.mean(),
                 optimal_class_among_candidates_pct=100 * s.has_opt.mean(), final_in_optimal_class_pct=100 * s.final_opt.mean(),
                 L_ratio=s.ratio.median(), L_ratio_p90=s.ratio.quantile(0.9))
        rows.append(r)
        lines.append(f"{int(K)} & {r['candidates']:.2f} & {r['classes']:.2f} & {r['optimal_class_among_candidates_pct']:.1f} & "
                     f"{r['final_in_optimal_class_pct']:.1f} & {r['L_ratio']:.3f} & {r['L_ratio_p90']:.3f} \\\\")
    H = pd.DataFrame(rows)
    W.csv("table12_homotopy", H, index=False)
    W.tex("t_homotopy", lines)
    q = d.drop_duplicates(KEY)
    S["E7"] = dict(n_queries=len(q), holes_median=float(q.n_obstacles.median()), holes_max=int(q.n_obstacles.max()),
                   table=H.set_index("K").to_dict(orient="index"))


# --------------------------------------------------------------------------- revision 1
EPS = (0.01, 0.02, 0.05, 0.10)


def epsilon(raw, W, S, order):
    """Empirical epsilon-optimality in E1: share of the queries with a valid path of length <= (1+eps) L*."""
    d = pd.read_csv(raw / "e1.csv")
    d["valid"] = valid(d)
    rows, lines = [], []
    for k in order:
        s = d[d.algo == k]
        r = s.relacion_optima.where(s.valid)
        row = dict(planner=NAMES[k], n=len(s), **{f"within_{int(round(100 * e))}pct": 100 * float((r <= 1 + e).mean())
                                                  for e in EPS}, max_ratio=float(r.max()))
        rows.append(row)
        lines.append(("\\textbf{TVE (proposed)}" if k == "tve" else cite_name(k)) + " & " +
                     " & ".join(f"{row[f'within_{int(round(100 * e))}pct']:.1f}" for e in EPS) +
                     f" & {row['max_ratio']:.3f} \\\\")
    E = pd.DataFrame(rows)
    W.csv("table06_epsilon_optimality", E, index=False)
    W.tex("t_epsilon", lines)
    S["epsilon"] = E.set_index("planner").to_dict(orient="index")
    return E


def e8(raw, W, S):
    """One-at-a-time sensitivity of the parameters of TVE."""
    d = pd.read_csv(raw / "e8.csv")
    d["valid"] = valid(d)
    labels = {"candidates": "$K$", "weight_alpha": "$\\alpha$", "penalty": "$\\beta$",
              "clearance_extra": "$\\delta$ [m]", "segment_size": "$\\ell_g$ [m]"}
    rows, lines = [], []
    # queries without reference optimum (the visibility graph found no path) are excluded from the ratios
    has_ref = d.groupby(KEY).relacion_optima.apply(lambda x: bool(x.notna().any()))
    no_ref = int((~has_ref).sum())
    d = d.join(has_ref.rename("has_ref"), on=KEY)
    for par, values in TVE_CFG["sensitivity_E8"].items():
        dp = d[d.parameter == par]
        base = dp[dp.is_default.astype(bool)].set_index(KEY)
        first = True
        for v in values:
            s = dp[np.isclose(dp.value.astype(float), float(v))]
            o = s[s.valid]
            rel = (s.set_index(KEY).t_ms / base.t_ms).median()
            dflt = bool(s.is_default.astype(bool).all())
            r = dict(parameter=par, value=v, default=dflt, valid_pct=100 * s.valid.mean(),
                     L_ratio=o.relacion_optima.median(), L_ratio_p90=o.relacion_optima.quantile(0.9),
                     within_2pct=100 * float((s[s.has_ref].relacion_optima.where(s[s.has_ref].valid) <= 1.02).mean()),
                     time_ms=o.t_ms.median(), relative_time=rel, ackermann_pct=100 * o.factible_ackermann.mean())
            rows.append(r)
            vtxt = f"{v:g}" + ("$^{*}$" if dflt else "")
            lines.append(f"{labels[par] if first else ''} & {vtxt} & {r['valid_pct']:.1f} & {r['L_ratio']:.3f} & "
                         f"{r['L_ratio_p90']:.3f} & {r['within_2pct']:.1f} & {rel:.2f} & {r['ackermann_pct']:.1f} \\\\")
            first = False
        lines.append("\\midrule")
    lines = lines[:-1]
    Sd = pd.DataFrame(rows)
    W.csv("table13_sensitivity", Sd, index=False)
    W.tex("t_sensitivity", lines)
    S["E8"] = dict(n_queries=int(d.groupby(KEY).ngroups), n_maps=int(d.drop_duplicates(["scene", "level", "seed"]).shape[0]),
                   n_queries_without_reference=no_ref, table=Sd.to_dict(orient="records"))
    return Sd


def e9(raw, W, S):
    """Narrow passages: crossing rate and length by passage width, geometry and planner."""
    d = pd.read_csv(raw / "e9.csv")
    d["valid"] = valid(d)
    q = ["kind", "gap_px", "seed"]
    best = d[d.valid].groupby(q).longitud_m.min().rename("L_best")
    d = d.join(best, on=q)
    d["L_over_best"] = d.longitud_m / d.L_best
    d["through_ok"] = d.through_passage.astype(bool) & d.valid
    g = d.groupby(["algo", "kind", "gap_px"])
    T = pd.DataFrame({"valid_pct": 100 * g.valid.mean(), "through_pct": 100 * g.through_ok.mean(),
                      "L_over_best_median": g.L_over_best.median(), "time_ms_median": g.tiempo_ms.median()}).reset_index()
    T.insert(1, "planner", T.algo.map(lambda k: NAMES.get(k, k)))
    W.csv("table14_narrow_passages", T, index=False)
    planners = [k for k in ["tve", "teselado", "mvdrm", "g2vd", "theta", "astar", "rrt_star", "prm", "visibility"]
                if k in set(d.algo)]
    gaps = sorted(d.gap_px.unique())
    lines = []
    tot = d.groupby(["algo", "gap_px"]).through_ok.mean().unstack() * 100
    for k in planners:
        lines.append(("\\textbf{TVE (proposed)}" if k == "tve" else cite_name(k)) + " & " +
                     " & ".join(f"{tot.loc[k, gp]:.0f}" for gp in gaps) + " \\\\")
    W.tex("t_narrow", lines)
    mpp = float(d.mpp.iloc[0])
    S["E9"] = dict(n_queries=int(d.groupby(q).ngroups), gaps_px=[int(x) for x in gaps], mpp=mpp,
                   through_pct=tot.to_dict(orient="index"),
                   tve_valid_pct=100 * float(d[d.algo == "tve"].valid.mean()),
                   tve_L_over_best_median=float(d[(d.algo == "tve") & d.valid].L_over_best.median()),
                   tve_L_over_best_max=float(d[(d.algo == "tve") & d.valid].L_over_best.max()))
    return T


def e10(raw, W, S):
    """Measured indicators of the Voronoi-based planners of Table 1 on the queries of E1."""
    d = pd.read_csv(raw / "e10.csv")
    d["valid"] = valid(d)
    d["gt_safe"] = d["gt_safe"].fillna(False).astype(bool)
    rows, lines = [], []
    for k in ["teselado", "mvdrm", "g2vd", "tve"]:
        s = d[d.algo == k]
        o = s[s.valid]
        r = dict(planner=NAMES.get(k, k), valid_pct=100 * s.valid.mean(), L_ratio=o.relacion_optima.median(),
                 L_ratio_p90=o.relacion_optima.quantile(0.9), time_ms=o.tiempo_ms.median(),
                 clearance_m=o.holgura_min_m.median(), sharp_turns=o.giros_bruscos.median(),
                 ackermann_pct=100 * o.factible_ackermann.mean())
        rows.append(r)
        name = {"teselado": "WDT", "mvdrm": "MVDRM", "g2vd": "G$^2$VD", "tve": "TVE"}[k]
        lines.append(f"{name} & {r['valid_pct']:.1f} & {r['L_ratio']:.3f} & {r['L_ratio_p90']:.3f} & {r['time_ms']:.0f} & "
                     f"{r['clearance_m']:.2f} & {r['sharp_turns']:.0f} & {r['ackermann_pct']:.1f} \\\\")
    R = pd.DataFrame(rows)
    W.csv("table01b_voronoi_planners", R, index=False)
    W.tex("t_related_measured", lines)
    S["E10"] = dict(n_queries=int(d.groupby(KEY).ngroups), table=R.set_index("planner").to_dict(orient="index"))
    return R


def r1_checks(raw, W, S):
    """Revision 1: reference optimum, pruning of candidates and the 12 maps of the skeleton correction."""
    out = {}
    f = raw / "r1_pruning_check.csv"
    if f.exists():
        d = pd.read_csv(f)
        gain = (d.L_paper_px - d.L_no_pruning_px) / d.L_paper_px
        ch = gain[gain.abs() > 1e-9]
        out["pruning"] = dict(n=len(d), changed=int(len(ch)), shorter=int((ch > 0).sum()), longer=int((ch < 0).sum()),
                              max_gain_pct=100 * float(ch.max()) if len(ch) else 0.0,
                              median_gain_pct=100 * float(ch.median()) if len(ch) else 0.0,
                              time_increase_pct=100 * float((d.t_no_pruning_ms / d.t_paper_ms).median() - 1))
    f = raw / "r1_reference_check.csv"
    if f.exists():
        r = pd.read_csv(f)
        e1 = pd.read_csv(raw / "e1.csv")
        e1["valid"] = valid(e1)
        excess = (r.L_ref_default_px - r.L_ref_tight_px) / r.L_ref_tight_px
        ref = dict(n=len(r), tight_shorter=int((excess > 1e-9).sum()), tight_longer=int((excess < -1e-9).sum()),
                   tight_longer_median_pct=100 * float((-excess[excess < -1e-9]).median()))
        r = r.set_index(KEY)
        # empirical bias of L*: shortest valid path returned by any of the fifteen planners
        Lall = (e1[e1.valid].longitud_m / e1[e1.valid].mpp)
        Lbest = e1[e1.valid].assign(L=Lall).groupby(KEY).L.min().reindex(r.index)
        gap = (r.L_ref_default_px - Lbest) / r.L_ref_default_px
        ref.update(best_below=int((gap > 1e-9).sum()), best_gap_median_pct=100 * float(gap[gap > 1e-9].median()),
                   best_gap_max_pct=100 * float(gap.max()), best_gap_p90_all_pct=100 * float(gap.clip(lower=0).quantile(0.9)))
        below = {}
        for k in ["tve", "rrt_star", "theta", "topological", "teselado", "astar"]:
            s = e1[(e1.algo == k) & e1.valid].set_index(KEY)
            Lpx = (s.longitud_m / s.mpp).reindex(r.index)
            below[k] = dict(below_default=int((Lpx < r.L_ref_default_px * (1 - 1e-9)).sum()),
                            below_tight=int((Lpx < r.L_ref_tight_px * (1 - 1e-9)).sum()),
                            min_ratio_default=float((Lpx / r.L_ref_default_px).min()),
                            min_ratio_tight=float((Lpx / r.L_ref_tight_px).min()),
                            median_ratio_tight=float((Lpx / r.L_ref_tight_px).median()),
                            within_2pct_tight=100 * float(((Lpx / r.L_ref_tight_px) <= 1.02).mean()))
        ref["planners"] = below
        out["reference"] = ref
    f = raw / "r1_skeleton_correction_maps.csv"
    if f.exists():
        m = pd.read_csv(f)
        q = pd.read_csv(raw / "r1_skeleton_correction_queries.csv")
        lines = []
        for r_ in m.itertuples():
            k = (q.experiment == r_.experiments.split(";")[0]) & (q.scene == r_.scene) & (q.seed == r_.seed)
            if r_.experiments != "E4":
                k &= (q.level == r_.level)
            s = q[k]
            lvl = LEVEL_EN.get(r_.level, r_.level)
            if s.ratio_before.notna().any():
                rb, ra = f"{s.ratio_before.max():.3f}", f"{s.ratio_after.max():.3f}"
            else:
                rb = ra = "--"
            lab = scene_labels().get(r_.scene, r_.scene)
            lines.append(f"{r_.experiments} & {lab} & {lvl} & {r_.seed} & {r_.size} & {int(r_.nodes_obstacles)} / "
                         f"{int(r_.nodes_contours)} & {rb} & {ra} \\\\")
        W.tex("t_skeleton_maps", lines)
        W.csv("tableB1_skeleton_correction_maps", m, index=False)
        out["skeleton"] = dict(n_maps=len(m), n_scanned=455)
    S["revision1_checks"] = out
    return out


# --------------------------------------------------------------------------- numbers quoted in the text
def numbers(T, rk, LV, S, W):
    e1, e2, e3, e4, e5, e7 = S["E1"], S["E2"], S["E3"], S["E4"], S["E5"], S["E7"]
    t = T
    n = {
        "NMaps": e1["n_maps"], "NQueries": e1["n_queries"], "NRuns": e1["n_runs"],
        "TveSucc": f"{t.loc['tve', 'valid_pct']:.1f}", "TveL": f"{t.loc['tve', 'L_ratio_median']:.3f}",
        "TveLp": f"{t.loc['tve', 'L_ratio_p90']:.3f}", "TveT": f"{t.loc['tve', 'time_ms_median']:.0f}",
        "WdtL": f"{t.loc['teselado', 'L_ratio_median']:.3f}", "WdtLp": f"{t.loc['teselado', 'L_ratio_p90']:.3f}",
        "WdtT": f"{t.loc['teselado', 'time_ms_median']:.0f}", "VisT": f"{t.loc['visibility', 'time_ms_median']:.0f}",
        "RrtsT": f"{t.loc['rrt_star', 'time_ms_median']:.0f}", "RrtsL": f"{t.loc['rrt_star', 'L_ratio_median']:.3f}",
        "TopoT": f"{t.loc['topological', 'time_ms_median']:.0f}", "TopoL": f"{t.loc['topological', 'L_ratio_median']:.3f}",
        "ThetaT": f"{t.loc['theta', 'time_ms_median']:.0f}", "ThetaL": f"{t.loc['theta', 'L_ratio_median']:.3f}",
        "AstarT": f"{t.loc['astar', 'time_ms_median']:.0f}", "AstarL": f"{t.loc['astar', 'L_ratio_median']:.3f}",
        "HybT": f"{t.loc['hybrid_astar', 'time_ms_median']:.0f}", "HybSucc": f"{t.loc['hybrid_astar', 'valid_pct']:.1f}",
        "TveAck": f"{t.loc['tve', 'ackermann_pct']:.1f}", "WdtAck": f"{t.loc['teselado', 'ackermann_pct']:.1f}",
        "TveGt": f"{t.loc['tve', 'gt_safe_pct']:.1f}",
        "FrChi": f"{e1['friedman_L']['chi2']:.1f}", "FrP": prel(e1["friedman_L"]["p"]),
        "FrN": e1["friedman_L"]["n"], "FrK": e1["friedman_L"]["k"],
        "FrChiT": f"{e1['friedman_time']['chi2']:.1f}", "FrPT": prel(e1["friedman_time"]["p"]),
        "TveRankL": f"{rk.loc['tve', 'rank_L']:.2f}", "TveRankT": f"{rk.loc['tve', 'rank_time']:.2f}",
        "PairN": e2["n"], "SpMed": f"{e2['speedup_median']:.2f}", "SpLo": f"{e2['speedup_ci'][0]:.2f}",
        "SpHi": f"{e2['speedup_ci'][1]:.2f}", "SpFaster": f"{e2['tve_faster_pct']:.1f}",
        "SpZ": f"{e2['time_wilcoxon']['z']:.2f}", "SpP": prel(e2["time_wilcoxon"]["p"]),
        "SpR": f"{abs(e2['time_wilcoxon']['r']):.2f}", "LZ": f"{e2['length_wilcoxon']['z']:.2f}",
        "LP": prel(e2["length_wilcoxon"]["p"]), "LR": f"{abs(e2['length_wilcoxon']['r']):.2f}",
        "Shorter": f"{e2['tve_shorter_pct']:.1f}",
        "PairTveT": f"{e2['time_ms_median']['tve']:.1f}", "PairWdtT": f"{e2['time_ms_median']['wdt']:.1f}",
        "PairTveL": f"{e2['L_ratio_median']['tve']:.3f}", "PairWdtL": f"{e2['L_ratio_median']['wdt']:.3f}",
        "PairTveLp": f"{e2['L_ratio_p90']['tve']:.3f}", "PairWdtLp": f"{e2['L_ratio_p90']['wdt']:.3f}",
        "PairTveAck": f"{e2['ackermann_pct']['tve']:.1f}", "PairWdtAck": f"{e2['ackermann_pct']['wdt']:.1f}",
        "TessTve": f"{e2['stage_times_ms']['TVE']['tessellation']:.1f}",
        "TessWdt": f"{e2['stage_times_ms']['WDT']['tessellation']:.1f}",
        "RedWdt": f"{e2['stage_times_ms']['WDT']['midpoint_reduction']:.1f}",
        "FunTve": f"{e2['stage_times_ms']['TVE']['funnel_and_validation']:.1f}",
        "SmWdt": f"{e2['stage_times_ms']['WDT']['smoothing']:.1f}", "SmTve": f"{e2['stage_times_ms']['TVE']['smoothing']:.1f}",
        "SkWdt": f"{e2['stage_times_ms']['WDT']['skeleton']:.1f}", "SkTve": f"{e2['stage_times_ms']['TVE']['skeleton']:.1f}",
        "MqTveFirst": f"{e3['tve']['first_ms']:.0f}", "MqTveRest": f"{e3['tve']['next_ms']:.1f}",
        "MqWdtRest": f"{e3['teselado']['next_ms']:.1f}", "MqAstar": f"{e3['astar']['next_ms']:.1f}",
        "MqTveTot": f"{e3['tve']['total_s']:.2f}", "MqWdtTot": f"{e3['teselado']['total_s']:.2f}",
        "MqVisTot": f"{e3['visibility']['total_s']:.1f}", "MqThetaTot": f"{e3['theta']['total_s']:.1f}",
        "MqAstarTot": f"{e3['astar']['total_s']:.2f}",
        "KTve": f"{e4['slopes'].get('tve', np.nan):.2f}", "KWdt": f"{e4['slopes'].get('teselado', np.nan):.2f}",
        "KVis": f"{e4['slopes'].get('visibility', np.nan):.2f}", "KTheta": f"{e4['slopes'].get('theta', np.nan):.2f}",
        "KAstar": f"{e4['slopes'].get('astar', np.nan):.2f}",
        "KTessTve": f"{e4['slopes'].get('tess_tve', np.nan):.2f}", "KTessWdt": f"{e4['slopes'].get('tess_teselado', np.nan):.2f}",
        "TessBigTve": f"{e4['tess_ms_largest'].get('tve', np.nan):.0f}",
        "TessBigWdt": f"{e4['tess_ms_largest'].get('teselado', np.nan):.0f}",
        "TotBigTve": f"{e4['time_ms_largest'].get('tve', np.nan):.0f}", "TotBigWdt": f"{e4['time_ms_largest'].get('teselado', np.nan):.0f}",
        "TotBigAstar": f"{e4['time_ms_largest'].get('astar', np.nan):.0f}",
        "TotBigTheta": f"{e4['time_ms_largest'].get('theta', np.nan):.0f}",
        "DynN": e5["n"],
    }
    tab5 = e5["table"]
    for k, tag in (("tve", "Tve"), ("teselado", "Wdt"), ("astar", "Astar"), ("theta", "Theta"), ("visibility", "Vis")):
        v = tab5[NAMES[k]]
        n[f"Dyn{tag}"] = f"{v['time_ms_median']:.0f}" if k == "visibility" else f"{v['time_ms_median']:.1f}"
        if k in ("tve", "teselado", "astar"):
            n[f"Dyn{tag}R"] = f"{v['L_ratio']:.3f}"
    s7 = e7["table"]
    n.update({"HomN": e7["n_queries"], "HomObst": f"{e7['holes_median']:.0f}", "HomObstMax": e7["holes_max"]})
    for K, tag in ((1, "One"), (2, "Two"), (4, "Four"), (8, "Eight")):
        n[f"HomHas{tag}"] = f"{s7[K]['optimal_class_among_candidates_pct']:.1f}"
    n.update({"HomClsFour": f"{s7[4]['classes']:.2f}", "HomCandFour": f"{s7[4]['candidates']:.2f}",
              "HomClsEight": f"{s7[8]['classes']:.2f}", "HomCandEight": f"{s7[8]['candidates']:.2f}",
              "HomFinOne": f"{s7[1]['final_in_optimal_class_pct']:.1f}", "HomFinFour": f"{s7[4]['final_in_optimal_class_pct']:.1f}",
              "HomLpOne": f"{s7[1]['L_ratio_p90']:.3f}", "HomLpFour": f"{s7[4]['L_ratio_p90']:.3f}"})
    lv = LV.set_index("level")
    for L_, tag in (("low", "Low"), ("medium", "Med"), ("high", "High")):
        n[f"Iou{tag}"] = f"{lv.loc[L_, 'iou']:.3f}"
        n[f"GtAll{tag}"] = f"{lv.loc[L_, 'all_gt_safe_pct']:.1f}"
        n[f"TveGt{tag}"] = f"{lv.loc[L_, 'tve_gt_safe_pct']:.1f}"
    # revision 1
    ep = S.get("epsilon", {})
    if ep:
        tv = ep[NAMES["tve"]]
        n.update({"EpsOne": f"{tv['within_1pct']:.1f}", "EpsTwo": f"{tv['within_2pct']:.1f}",
                  "EpsFive": f"{tv['within_5pct']:.1f}", "EpsTen": f"{tv['within_10pct']:.1f}",
                  "EpsMax": f"{tv['max_ratio']:.3f}"})
        for k, tag in (("teselado", "Wdt"), ("theta", "Theta"), ("rrt_star", "Rrts"), ("astar", "Astar"),
                       ("topological", "Topo")):
            n[f"Eps{tag}Two"] = f"{ep[NAMES[k]]['within_2pct']:.1f}"
            n[f"Eps{tag}Five"] = f"{ep[NAMES[k]]['within_5pct']:.1f}"
    if "E8" in S:
        t8 = pd.DataFrame(S["E8"]["table"])

        def g8(par, val, col):
            return t8[(t8.parameter == par) & np.isclose(t8.value.astype(float), val)][col].iloc[0]
        n.update({"SensN": S["E8"]["n_queries"], "SensMaps": S["E8"]["n_maps"],
                  "SensNoRef": S["E8"]["n_queries_without_reference"],
                  "SensDefL": f"{g8('candidates', 4, 'L_ratio'):.3f}", "SensDefLp": f"{g8('candidates', 4, 'L_ratio_p90'):.3f}",
                  "SensKOneLp": f"{g8('candidates', 1, 'L_ratio_p90'):.3f}", "SensKEightLp": f"{g8('candidates', 8, 'L_ratio_p90'):.3f}",
                  "SensKEightT": f"{g8('candidates', 8, 'relative_time'):.2f}", "SensKOneT": f"{g8('candidates', 1, 'relative_time'):.2f}",
                  "SensBetaLowLp": f"{g8('penalty', 1.2, 'L_ratio_p90'):.3f}", "SensAlphaOneLp": f"{g8('weight_alpha', 1.0, 'L_ratio_p90'):.3f}",
                  "SensDeltaZeroL": f"{g8('clearance_extra', 0.0, 'L_ratio'):.3f}",
                  "SensDeltaZeroAck": f"{g8('clearance_extra', 0.0, 'ackermann_pct'):.1f}",
                  "SensDeltaMaxL": f"{g8('clearance_extra', 0.2, 'L_ratio'):.3f}",
                  "SensDeltaMaxAck": f"{g8('clearance_extra', 0.2, 'ackermann_pct'):.1f}",
                  "SensDefAck": f"{g8('candidates', 4, 'ackermann_pct'):.1f}"})
        others = t8[~t8.parameter.isin(["clearance_extra"]) & ~((t8.parameter == "candidates") & (t8.value == 1))]
        n["SensSpreadL"] = f"{100 * (others.L_ratio.max() - others.L_ratio.min()):.2f}"
    if "E9" in S:
        e9_ = S["E9"]
        n.update({"NarrowN": e9_["n_queries"], "NarrowTveValid": f"{e9_['tve_valid_pct']:.0f}",
                  "NarrowTveL": f"{e9_['tve_L_over_best_median']:.3f}", "NarrowMpp": f"{e9_['mpp']:.2f}"})
        for k, tag in (("tve", "Tve"), ("teselado", "Wdt"), ("visibility", "Vis"), ("theta", "Theta"), ("astar", "Astar"),
                       ("g2vd", "Gvd"), ("mvdrm", "Mvdrm"), ("rrt_star", "Rrts"), ("prm", "Prm")):
            if k in e9_["through_pct"]:
                n[f"Narrow{tag}One"] = f"{e9_['through_pct'][k][1]:.0f}"
                n[f"Narrow{tag}Two"] = f"{e9_['through_pct'][k][2]:.0f}"
    chk = S.get("revision1_checks", {})
    if "pruning" in chk:
        p_ = chk["pruning"]
        n.update({"PruneChanged": p_["changed"], "PruneMaxGain": f"{p_['max_gain_pct']:.1f}",
                  "PruneMedGain": f"{p_['median_gain_pct']:.2f}", "PruneTime": f"{p_['time_increase_pct']:.0f}"})
    if "reference" in chk:
        rf = chk["reference"]
        pl = rf["planners"]
        n.update({"RefTightLonger": rf["tight_longer"], "RefTightShorter": rf["tight_shorter"],
                  "RefTightLongerMed": f"{rf['tight_longer_median_pct']:.1f}",
                  "RefBestBelow": rf["best_below"], "RefBestGapMed": f"{rf['best_gap_median_pct']:.2f}",
                  "RefBestGapMax": f"{rf['best_gap_max_pct']:.1f}",
                  "RefTveBelow": pl["tve"]["below_default"], "RefTveMin": f"{pl['tve']['min_ratio_default']:.3f}",
                  "RefTopoBelow": pl["topological"]["below_default"], "RefRrtsBelow": pl["rrt_star"]["below_default"],
                  "RefThetaBelow": pl["theta"]["below_default"]})
    if "skeleton" in chk:
        n["SkelMaps"] = chk["skeleton"]["n_maps"]
    if "E10" in S:
        t10 = S["E10"]["table"]
        for k, tag in (("tve", "Tve"), ("teselado", "Wdt"), ("mvdrm", "Mvdrm"), ("g2vd", "Gvd")):
            v = t10[NAMES.get(k, k)]
            n.update({f"Vor{tag}L": f"{v['L_ratio']:.3f}", f"Vor{tag}Lp": f"{v['L_ratio_p90']:.3f}",
                      f"Vor{tag}T": f"{v['time_ms']:.0f}", f"Vor{tag}Ack": f"{v['ackermann_pct']:.1f}",
                      f"Vor{tag}Valid": f"{v['valid_pct']:.1f}", f"Vor{tag}Clr": f"{v['clearance_m']:.2f}"})
    W.tex("numbers", ["% Generated by statistics/analyze.py - do not edit"] +
          [f"\\newcommand{{\\n{k}}}{{{v}}}" for k, v in n.items()])
    return n


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "results" / "tables"))
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    W = Writer(out)
    S = {}
    T, rk, LV = e1(raw, W, S)
    epsilon(raw, W, S, list(T.index))
    e2(raw, W, S)
    e3(raw, W, S)
    e4(raw, W, S)
    e5(raw, W, S)
    e6(raw, W, S)
    e7(raw, W, S)
    for f in (e8, e9, e10):                       # revision 1
        if (raw / f"{f.__name__}.csv").exists():
            f(raw, W, S)
    r1_checks(raw, W, S)
    S["numbers_in_text"] = numbers(T, rk, LV, S, W)
    (out / "statistics_summary.json").write_text(json.dumps(S, indent=2, default=_json_default), encoding="utf-8")
    print(f"tables and statistics written to {out}")


if __name__ == "__main__":
    main()
