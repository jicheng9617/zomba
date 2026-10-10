"""Figures and summary tables for the theory-vs-practice pilot (reads ./results/*.json)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import bounds as bd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
FIG.mkdir(exist_ok=True)

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# one colour per algorithm family, the same in every figure
FAMILY_COLOR = {"UCB": SLOT[0], "KL-UCB": SLOT[1], "TS": SLOT[2], "MOSS": SLOT[3], "theory": SLOT[4],
                "ETC": SLOT[5], "eps": SLOT[6], "Greedy": SLOT[7]}


def color_of(name):
    if "c=3" in name:
        return FAMILY_COLOR["theory"]
    for key, col in [("KL-UCB", "KL-UCB"), ("TS", "TS"), ("MOSS", "MOSS"), ("ETC", "ETC"),
                     ("eps", "eps"), ("Greedy", "Greedy"), ("UCB", "UCB")]:
        if key in name:
            return FAMILY_COLOR[col]
    return INK2


plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK2,
})


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def ref_line(ax, x, y, label, color=INK, **kw):
    ax.plot(x, y, color=color, lw=1.4, ls=(0, (5, 3)), label=label, **kw)


def end_label(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", va="center",
                fontsize=8, color=INK2)


# ============================================================================ E1
def fig_e1(out):
    r = load("e1")
    if r is None:
        return
    cps = np.array(r["checkpoints"], float)
    keep = cps >= 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    titles = {"I1": "K=2, mu=(0.9, 0.8)", "I2": "K=10, low means (0.1, 0.05x3, 0.02x3, 0.01x3)"}
    rows = []
    for ax, (iname, e) in zip(axes, r["instances"].items()):
        x = cps[keep]
        for aname, a in e["algs"].items():
            m, se = np.array(a["mean"])[keep], np.array(a["se"])[keep]
            y = m / np.log(x)
            ax.plot(x, y, color=color_of(aname), label=aname)
            ax.fill_between(x, (m - 1.96 * se) / np.log(x), (m + 1.96 * se) / np.log(x),
                            color=color_of(aname), alpha=0.12, lw=0)
        ub = np.array(e["ucb1_bound"])[keep] / np.log(x)
        ref_line(ax, x, ub, "UCB1 bound (Auer et al. 2002)", color=FAMILY_COLOR["UCB"])
        ref_line(ax, x, np.full_like(x, e["C_LR"]), "Lai-Robbins constant")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("horizon n")
        ax.set_ylabel("regret / log n")
        ax.set_title(f"{iname}: {titles[iname]}")
        for aname, a in e["algs"].items():
            rows.append((iname, aname, a["mean"][-1], 1.96 * a["se"][-1],
                         a["mean"][-1] / (e["C_LR"] * np.log(cps[-1])),
                         a["mean"][-1] / e["ucb1_bound"][-1]))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncols=6)
    fig.suptitle("E1  Gap-dependent theory: regret/log n vs. the UCB1 bound and the Lai-Robbins constant "
                 "(Bernoulli, 300 runs, 95% CI)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.savefig(FIG / "e1_gap_dependent.png", dpi=160)
    plt.close(fig)

    out.append("### E1 gap-dependent bounds (n = 10^6)\n")
    out.append("| instance | algorithm | R_n | 95% CI +- | R_n / (C_LR log n) | R_n / UCB1-bound |")
    out.append("|---|---|---:|---:|---:|---:|")
    for row in rows:
        out.append(f"| {row[0]} | {row[1]} | {row[2]:.1f} | {row[3]:.1f} | {row[4]:.2f} | {row[5]:.3f} |")
    rb = r["random_instances"]
    u = np.array(rb["ucb1_over_bound"])
    out.append(f"\nRandom instances (K=10, mu ~ U(0,1)^10, n = {rb['n']}, {rb['M']} instances): "
               f"UCB1 regret / UCB1 bound median {np.median(u):.3f} (IQR {np.quantile(u, .25):.3f}-{np.quantile(u, .75):.3f}); "
               f"bound >= trivial n*Delta_max on {100 * rb['bound_vacuous_frac']:.0f}% of instances; "
               f"TS / (C_LR log n) median {np.median(rb['ts_over_LR']):.2f}, "
               f"KL-UCB / (C_LR log n) median {np.median(rb['klucb_over_LR']):.2f}.\n")


# ============================================================================ E2
def fig_e2(out):
    r = load("e2")
    if r is None:
        return
    K, ns, gaps = r["K"], np.array(r["ns"], float), np.array(r["gaps"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.0), layout="constrained")
    table = []
    for aname, a in r["algs"].items():
        m = np.array(a["mean"])          # (G, len(ns))
        se = np.array(a["se"])
        jw = m.argmax(0)
        worst = m[jw, np.arange(len(ns))]
        worst_se = se[jw, np.arange(len(ns))]
        a1.plot(ns, worst / np.sqrt(K * ns), color=color_of(aname), marker="o", ms=4.5,
                markeredgecolor=SURF, markeredgewidth=1.2, label=aname)
        slope = np.polyfit(np.log(ns), np.log(worst), 1)[0]
        table.append((aname, worst, worst_se, gaps[jw], slope))
        j = list(r["ns"]).index(10000)
        a2.plot(gaps, m[:, j], color=color_of(aname), label=aname)
    ref_line(a1, ns, bd.moss_minimax(ns, K, 1.0 * (K - 1)) / np.sqrt(K * ns), "MOSS bound 39 sqrt(Kn) + sum gaps",
             color=FAMILY_COLOR["MOSS"])
    ref_line(a1, ns, bd.ucb_delta_minimax(ns, K, 1.0 * (K - 1)) / np.sqrt(K * ns),
             "UCB bound 8 sqrt(Kn log n) + 3 sum gaps", color=FAMILY_COLOR["UCB"])
    ref_line(a1, ns, bd.minimax_lower(ns, K) / np.sqrt(K * ns), "lower bound sqrt((K-1)n)/27")
    ref_line(a1, ns, ns / np.sqrt(K * ns), "trivial bound n (means in [0,1])", color=MUTED)
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel("horizon n")
    a1.set_ylabel("worst-case regret / sqrt(K n)")
    a1.set_title("Worst case over the hard family, normalised by sqrt(Kn)")
    n_vac = 39.0 ** 2 * K  # 39 sqrt(Kn) >= n  <=>  n <= 1521 K
    a1.axvline(n_vac, color=MUTED, lw=1, ls=(0, (2, 2)))
    a1.annotate(f"MOSS bound > trivial\nfor n < {n_vac:,.0f}", (n_vac, 0.06), xytext=(-4, 0),
                textcoords="offset points", ha="right", fontsize=8, color=INK2)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncols=4, fontsize=8)
    j = list(r["ns"]).index(10000)
    a2.axvline(np.sqrt(K / 10000), color=MUTED, lw=1, ls=(0, (2, 2)))
    a2.annotate("sqrt(K/n)", (np.sqrt(K / 10000), 0.97), xycoords=("data", "axes fraction"),
                xytext=(3, 0), textcoords="offset points", fontsize=8, color=INK2)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("gap Delta (best arm Delta, other 9 arms 0)")
    a2.set_ylabel("expected regret at n = 10^4")
    a2.set_title("Regret profile over the gap, n = 10^4")
    fig.suptitle(f"E2  Empirical minimax regret vs. sqrt(Kn) theory (K={K}, unit-variance Gaussian, "
                 f"{r['runs']} runs per cell, 24 gaps)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.savefig(FIG / "e2_minimax.png", dpi=160)
    plt.close(fig)

    out.append("### E2 empirical worst-case regret (max over 24 gaps of the mean regret)\n")
    out.append("| algorithm | " + " | ".join(f"n={int(n)}" for n in ns) + " | fitted exponent b (R* ~ n^b) | worst gap at n=10^4 |")
    out.append("|---|" + "---:|" * (len(ns) + 2))
    for aname, worst, wse, dg, slope in table:
        cells = " | ".join(f"{w:.0f} ({w / np.sqrt(K * n):.2f})" for w, n in zip(worst, ns))
        out.append(f"| {aname} | {cells} | {slope:.2f} | {dg[list(r['ns']).index(10000)]:.3f} |")
    out.append("\nCells: worst-case regret (and that divided by sqrt(Kn)).")
    lb = bd.minimax_lower(ns, K)
    mb = bd.moss_minimax(ns, K, K - 1.0)
    ub = bd.ucb_delta_minimax(ns, K, K - 1.0)  # sum of gaps <= K - 1 for means in [0, 1]
    moss_w = [t for t in table if t[0] == "MOSS"][0][1]
    ucb_w = [t for t in table if t[0].startswith("UCB")][0][1]
    out.append("\n| n | lower bound | MOSS: bound / empirical | UCB(delta): bound / empirical | best empirical / lower bound |")
    out.append("|---:|---:|---:|---:|---:|")
    best_emp = np.min([t[1] for t in table], axis=0)
    for i, n in enumerate(ns):
        out.append(f"| {int(n)} | {lb[i]:.1f} | {mb[i]:.0f} / {moss_w[i]:.0f} = {mb[i] / moss_w[i]:.0f}x | "
                   f"{ub[i]:.0f} / {ucb_w[i]:.0f} = {ub[i] / ucb_w[i]:.0f}x | {best_emp[i] / lb[i]:.1f}x |")
    out.append("")

    r2 = load("e2b")
    if r2:
        out.append(f"E2b random search at n = {r2['n']} ({r2['M']} instances x 200 runs per set):\n")
        out.append("| algorithm | hard-family max | perturbed-family max | uniform U(0,1)^K max |")
        out.append("|---|---:|---:|---:|")
        for aname, a in r2["algs"].items():
            out.append(f"| {aname} | {a['family_max']:.0f} | {a['perturbed']['max']:.0f} | {a['uniform']['max']:.0f} |")
        out.append("")


# ============================================================================ E3
def fig_e3(out):
    r = load("e3")
    if r is None:
        return
    pols = list(r["settings"]["fixed arms"].keys())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained",
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(pols))
    for k_, (setting, col, off) in enumerate([("fixed arms", SLOT[0], -0.13), ("fresh contexts", SLOT[1], 0.13)]):
        s = r["settings"][setting]
        m = np.array([s[p]["mean"][-1] for p in pols])
        ci = np.array([1.96 * s[p]["se"][-1] for p in pols])
        a1.errorbar(m, y + off, xerr=ci, fmt="o", ms=6, color=col, mec=SURF, mew=1.2, elinewidth=1.4,
                    capsize=0, label=setting)
    a1.axvline(r["theory_bound"][-1], color=INK, lw=1.4, ls=(0, (5, 3)))
    a1.annotate(f"OFUL bound\n(Thm 3)\n{r['theory_bound'][-1]:,.0f}", (r["theory_bound"][-1], 0.6),
                xytext=(4, 0), textcoords="offset points", ha="left", fontsize=8, color=INK2)
    a1.axvline(2 * r["n"], color=MUTED, lw=1.2, ls=(0, (2, 2)))
    a1.annotate(f"trivial\n2n = {2 * r['n']:,}", (2 * r["n"], 0.6), xytext=(-4, 0), textcoords="offset points",
                ha="right", fontsize=8, color=INK2)
    a1.set_xlim(150, 1.2e5)
    a1.set_yticks(y, pols)
    a1.invert_yaxis()
    a1.set_xscale("log")
    a1.set_xlabel(f"regret at n = {r['n']:,} (mean, 95% CI)")
    a1.set_title(f"Regret, d={r['d']}, K={r['K']}, R={r['R']}, {r['runs']} runs")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=2)
    a1.grid(axis="y", visible=False)

    s = r["settings"]["fixed arms"]
    for key, col, off, lab in [("worst_ratio", SLOT[6], -0.13, "whole radius, Thm 2 (incl. sqrt(lam) S bias)"),
                               ("worst_noise_ratio", SLOT[2], 0.13, "noise term only, Thm 1")]:
        q50 = np.array([np.median(s[p][key]) for p in pols])
        q99 = np.array([np.quantile(s[p][key], 0.99) for p in pols])
        for i in range(len(pols)):
            a2.plot([q50[i], q99[i]], [i + off, i + off], color=col, lw=1.4, alpha=0.45)
        a2.plot(q99, y + off, "o", ms=6, color=col, mec=SURF, mew=1.2, label=lab, ls="none")
        a2.plot(q50, y + off, "o", ms=5, mfc=SURF, mec=col, mew=1.4, ls="none")
    a2.axvline(1.0, color=INK, lw=1.4, ls=(0, (5, 3)))
    a2.annotate("bound\n(exceeded with\nprob. <= delta)", (1.0, 5.5), xytext=(5, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8, color=INK2)
    a2.set_yticks(y, [""] * len(pols))
    a2.invert_yaxis()
    a2.set_xlim(0, 1.4)
    a2.set_xlabel("max over t: deviation / theoretical radius  (hollow median, filled 99th pct)")
    a2.set_title("How much of the confidence radius is used (fixed arms)")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=2)
    a2.grid(axis="y", visible=False)
    fig.suptitle("E3  Linear bandits: theory-tuned vs. practical exploration, and confidence-set slack",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.savefig(FIG / "e3_linear.png", dpi=160)
    plt.close(fig)

    out.append(f"### E3 linear bandit (d={r['d']}, K={r['K']}, n={r['n']}, R={r['R']}, delta={r['delta']}, "
               f"LinTS theory inflation v={r['v_theory']:.1f})\n")
    out.append("| policy | regret, fixed arms | regret, fresh contexts | Thm 2 slack median / q99 | noise-only (Thm 1) median / q99 | runs above 1 (Thm 2 / Thm 1) |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for p in pols:
        f_, c_ = r["settings"]["fixed arms"][p], r["settings"]["fresh contexts"][p]
        w, wn = np.array(f_["worst_ratio"]), np.array(f_["worst_noise_ratio"])
        out.append(f"| {p} | {f_['mean'][-1]:.0f} +- {1.96 * f_['se'][-1]:.0f} | {c_['mean'][-1]:.0f} +- {1.96 * c_['se'][-1]:.0f} | "
                   f"{np.median(w):.2f} / {np.quantile(w, .99):.2f} | {np.median(wn):.2f} / {np.quantile(wn, .99):.2f} | "
                   f"{int((w > 1).sum())} / {int((wn > 1).sum())} of {len(w)} |")
    out.append(f"\nTheory regret bound at n={r['n']}: {r['theory_bound'][-1]:,.0f} (trivial bound: n * max gap <= {2 * r['n']:,}).\n")

    rb = load("e3b")
    if rb:
        cs = rb["cs"]
        out.append(f"E3b robustness, OFUL with radius c * sqrt(beta_t), fixed arms, K={rb['K']}, n={rb['n']}, "
                   f"{rb['runs']} runs: regret relative to the best c\n")
        out.append("| d | R | " + " | ".join(f"c={c}" for c in cs) + " | best c | Thm 2 slack q99 (c=1) | noise-only q99 (c=1) |")
        out.append("|---:|---:|" + "---:|" * (len(cs) + 3))
        for row in rb["configs"]:
            reg = np.array(row["regret"])
            out.append(f"| {row['d']} | {row['R']} | " + " | ".join(f"{v / reg.min():.2f}" for v in reg)
                       + f" | {cs[int(reg.argmin())]} | {row['slack_q99'][0]:.2f} | {row['noise_q99'][0]:.2f} |")
        out.append("")


# ============================================================================ E4
def fig_e4(out):
    r = load("e4")
    if r is None:
        return
    fams = r["families"]
    generic = [a for a in next(iter(fams.values())).keys() if a not in ("KL-UCB (c=0)", "TS-Beta")]
    ratios = {a: [] for a in generic}
    per_fam = {}
    for fname, e in fams.items():
        M = np.array([e[a]["mean"] for a in generic])          # (A, instances)
        best = M.min(0)
        rat = M / np.maximum(best, 1e-9)
        per_fam[fname] = {a: rat[i] for i, a in enumerate(generic)}
        for i, a in enumerate(generic):
            ratios[a].extend(rat[i].tolist())
    taus = np.geomspace(1, 100, 400)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for a in generic:
        v = np.array(ratios[a])
        ax.plot(taus, [(v <= t).mean() for t in taus], color=color_of(a), label=a, drawstyle="steps-post")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("tau  (regret within tau x the best algorithm on that instance)")
    ax.set_ylabel("fraction of problem instances")
    ax.set_title(f"E4  Performance profile over {len(fams)} environment families x {r['M']} instances (n = {r['n']:,})")
    ax.legend(loc="lower right")
    fig.savefig(FIG / "e4_generality.png", dpi=160)
    plt.close(fig)

    out.append("### E4 generality: geometric-mean regret ratio to the per-instance best (generic algorithms)\n")
    out.append("| family | " + " | ".join(generic) + " |")
    out.append("|---|" + "---:|" * len(generic))
    for fname, d_ in per_fam.items():
        out.append(f"| {fname} | " + " | ".join(f"{np.exp(np.mean(np.log(d_[a]))):.2f}" for a in generic) + " |")
    out.append("| **all (geo-mean)** | " + " | ".join(f"**{np.exp(np.mean(np.log(ratios[a]))):.2f}**" for a in generic) + " |")
    out.append("| worst family | " + " | ".join(f"{max(np.exp(np.mean(np.log(d_[a]))) for d_ in per_fam.values()):.2f}" for a in generic) + " |")
    bern = [f for f in fams if "Bernoulli" in f]
    out.append("\nBernoulli families incl. Bernoulli-specific algorithms (mean regret):\n")
    algs = list(fams[bern[0]].keys())
    out.append("| family | " + " | ".join(algs) + " |")
    out.append("|---|" + "---:|" * len(algs))
    for f in bern:
        out.append(f"| {f} | " + " | ".join(f"{np.mean(fams[f][a]['mean']):.0f}" for a in algs) + " |")
    out.append("")


if __name__ == "__main__":
    lines = ["# Pilot results (auto-generated by make_figures.py)\n"]
    for f in (fig_e1, fig_e2, fig_e3, fig_e4):
        f(lines)
    (HERE / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
