"""eps_r = 2, five training concentrations, V2 from three seeds: V1 vs V2 against MD.

(a) Gamma = 2n/S_NN(k_1) vs c: MD (eps_r = 2, and eps_r = 7.5 for reference), V1, V2
(b) force-balance chi2 of every held-out run, and of the untrained c = 0.05 runs
(c) c = 0.04 p27, the strongest neutral well (6.5 kT): anion density (V2's worst run)
(d) c = 0.01 p14, a neutral Gaussian train at low c: cation density

    SIP_ROOT=<research root> python figs/eps2_first.py   (cwd = <root>/code)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.environ["SIP_ROOT"]
sys.path.insert(0, os.path.join(R, "code"))
import learn.data as D

L = os.path.join(R, "runs", "learn")
T = "c001_c002_c004_c006_c008"
V1 = os.path.join(L, f"v1_{T}_full")
V2S = [os.path.join(L, f"joint_v2_L1_C4_R128x128_H64_{T}_val3_full{s}") for s in ("", "_s1", "_s2")]
V2 = V2S[0]
GM75 = "/mnt/gs21/scratch/lyuliyao/salt_in_polymer/runs/learn/interpretation/gamma_md.json"
C_MD, C_V2, C_V1, C_REF = "#0b0b0b", "#2a78d6", "#eb6834", "#8a8984"
INK2, GRID = "#52514e", "#e4e3de"
TRAIN = [0.01, 0.02, 0.04, 0.06, 0.08]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "axes.labelcolor": "#1f1f1e", "xtick.color": INK2,
                     "ytick.color": INK2, "legend.frameon": False, "axes.titlesize": 9.5,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})


GK1 = json.load(open(os.path.join(L, "gamma_k1_full.json")))   # figs/gamma_k1.py, as paper Fig. 3


def gamma_model(d):
    """Gamma = 2 nbar / S_NN(k_1), the paper's definition (not the k -> 0 limit in metrics.json)."""
    return {float(c): v for c, v in GK1[os.path.basename(d)].items()}


def rows(d, key="heldout_rows", sub=None):
    f = os.path.join(d, sub, "metrics.json") if sub else os.path.join(d, "metrics.json")
    return json.load(open(f))[key]


fig, ax = plt.subplots(2, 2, figsize=(10, 7.4))
fig.subplots_adjust(hspace=0.42, wspace=0.26, left=0.07, right=0.98, top=0.90, bottom=0.08)

# ---------------------------------------------------------------- (a) Gamma
a = ax[0, 0]
gm = json.load(open(os.path.join(L, "interpretation", "gamma_md.json")))
cs = sorted(float(c) for c in gm)
g = [gm[f"{c:g}"]["shell1"]["Gamma"] for c in cs]
ge = [gm[f"{c:g}"]["shell1"]["Gamma_err"] for c in cs]
if os.path.exists(GM75):
    g75 = json.load(open(GM75))
    c75 = sorted(float(c) for c in g75 if float(c) <= 0.08)
    a.errorbar(c75, [g75[f"{c:g}"]["shell1"]["Gamma"] for c in c75],
               [g75[f"{c:g}"]["shell1"]["Gamma_err"] for c in c75], fmt="o", mfc="white",
               mec=C_REF, ecolor=C_REF, ms=6, lw=1, capsize=0, label="MD, $\\varepsilon_r$ = 7.5 (reference)")
g1 = gamma_model(V1)
g2s = [gamma_model(d) for d in V2S]
xs = sorted(g2s[0])
g2m = np.array([np.mean([g[c] for g in g2s]) for c in xs])
g2lo = np.array([min(g[c] for g in g2s) for c in xs]); g2hi = np.array([max(g[c] for g in g2s) for c in xs])
a.plot(xs, [g1[c] for c in xs], "-s", color=C_V1, ms=6, lw=2, label="V1 (pair kernel)")
a.fill_between(xs, g2lo, g2hi, color=C_V2, alpha=0.25, lw=0)
a.plot(xs, g2m, "-D", color=C_V2, ms=6, lw=2, label="V2 (functional), 3 seeds: mean, range")
a.errorbar(cs, g, ge, fmt="o", color=C_MD, ms=7, lw=1.2, capsize=0, zorder=5, label="MD, $\\varepsilon_r$ = 2")
a.axvspan(0.047, 0.053, color=GRID, zorder=0)
a.text(0.05, 17, "c = 0.05\nnot trained", ha="center", va="top", fontsize=8, color=INK2)
a.set_yscale("log")
a.set_xlabel("$c_{LJ}$")
a.set_ylabel("$\\Gamma = 2n/S_{NN}(k_1)$")
a.set_title("(a) $\\Gamma$ at $k_1 = 2\\pi/L$ (never trained on)")
a.legend(fontsize=8, loc="lower right")
a.set_ylim(0.08, 20)

# ---------------------------------------------------------------- (b) chi2 per run
b = ax[0, 1]
groups = TRAIN + [0.05]
def run_chi2(d, c):
    rr = rows(d, "transfer_rows", "predict_c005") if c == 0.05 else \
        [r for r in rows(d) if abs(r["conc"] - c) < 1e-9]
    return rr, np.array([r["chi2"] for r in rr])


for lab, ds, col, mk, dx in (("V1", [V1], C_V1, "s", -0.12), ("V2 (seed mean)", V2S, C_V2, "D", 0.12)):
    for i, c in enumerate(groups):
        rr, _ = run_chi2(ds[0], c)
        ys = np.array([run_chi2(d, c)[1] for d in ds])
        y = ys.mean(0)
        if len(ds) > 1:
            xx = i + dx + np.linspace(-0.06, 0.06, len(y)) if len(y) > 1 else np.array([i + dx])
            b.vlines(xx, ys.min(0), ys.max(0), color=col, lw=1, alpha=0.6, zorder=2)
        x = i + dx + np.linspace(-0.06, 0.06, len(y)) if len(y) > 1 else [i + dx]
        b.scatter(x, y, s=22, marker=mk, color=col, edgecolor="white", linewidth=0.6, zorder=3,
                  label=lab if i == 0 else None)
        for r, xx, yy in zip(rr, x, y):
            if yy > 5:
                b.annotate(f"{r['tag']}", (xx, yy), xytext=(6, 0), textcoords="offset points",
                           fontsize=7.5, color=INK2, va="center")
b.axhline(1, color=INK2, lw=0.8, ls="--")
b.text(-0.45, 1.08, "noise level", fontsize=7.5, color=INK2, va="bottom")
b.axvline(len(TRAIN) - 0.5, color=GRID, lw=1)
b.set_yscale("log")
b.set_xticks(range(len(groups)))
b.set_xticklabels([f"{c:g}\nheld-out" for c in TRAIN] + ["0.05\nuntrained c"])
b.set_ylabel("force-balance $\\chi^2$ per bin\n(model force on the MD density vs YBG)")
b.set_title("(b) force $\\chi^2$ of every held-out run, and of the untrained c")
b.set_ylim(top=400)
b.legend(fontsize=8, loc="upper left", ncol=2)


# ---------------------------------------------------------------- (c, d) profiles
sps, runs = D.load_all(concs=[0.01, 0.04])
p2s = [np.load(os.path.join(d, "heldout.npz")) for d in V2S]
byk = {f"c{r.conc:g}_{r.tag}": r for r in runs}
p1 = np.load(os.path.join(V1, "heldout.npz"))
p2 = np.load(os.path.join(V2, "heldout.npz"))


def profile(axx, key, title, sp_i=0, lx=0.5):
    r = byk[key]
    sp = sps[r.conc]
    n_md, sg = np.asarray(r.n)[sp_i], np.asarray(r.sig_n)[sp_i]
    z = (np.arange(len(n_md)) + 0.5) * sp.L / len(n_md) if r.fold_q in (0, 1) else \
        (np.arange(len(n_md)) + 0.5) * sp.L / r.fold_q / len(n_md)
    nb = sp.n_pairs / sp.volume
    y1, y2 = np.asarray(p1[key])[sp_i], np.asarray(p2[key])[sp_i]
    if len(y1) != len(n_md):          # predictions on the full box, MD folded
        y1 = y1[:len(n_md)]; y2 = y2[:len(n_md)]
    axx.fill_between(z, (n_md - 2 * sg) / nb, (n_md + 2 * sg) / nb, color=GRID, lw=0, label="MD ±2σ")
    axx.plot(z, n_md / nb, color=C_MD, lw=1.2, label="MD")
    axx.plot(z, y1 / nb, color=C_V1, lw=2, label="V1")
    for j, pp in enumerate(p2s):
        yy = np.asarray(pp[key])[sp_i][:len(n_md)]
        axx.plot(z, yy / nb, color=C_V2, lw=2 if j == 0 else 1, ls="--", alpha=1 if j == 0 else 0.6,
                 label="V2 (3 seeds)" if j == 0 else None)
    lab = "cation" if sp_i == 0 else "anion"
    rv1 = {f"c{x['conc']:g}_{x['tag']}": x for x in rows(V1)}[key]
    v2l = [100 * {f"c{x['conc']:g}_{x['tag']}": x for x in rows(d)}[key]["profile"][lab]["rel_l2"] for d in V2S]
    axx.set_title(f"{title}\n      {lab} profile rel. L2: V1 {100 * rv1['profile'][lab]['rel_l2']:.0f}%, "
                  f"V2 {np.mean(v2l):.1f}% ({min(v2l):.1f}-{max(v2l):.1f})", fontsize=9.5)
    axx.set_xlabel("z / σ" + (f"  (folded over period {sp.L / r.fold_q:.2f} σ)" if r.fold_q > 1 else ""))
    axx.set_ylabel("$n_+(z)/\\bar n$" if sp_i == 0 else "$n_-(z)/\\bar n$")
    lo, hi = axx.get_ylim()
    axx.set_ylim(lo, hi + 0.22 * (hi - lo))
    axx.legend(fontsize=7.5, loc="upper center", ncol=4, columnspacing=1.2, handlelength=1.8)


profile(ax[1, 0], "c0.04_p27", "(c) c = 0.04, p27: strongest neutral well (6.5 kT)", sp_i=1, lx=0.31)
profile(ax[1, 1], "c0.01_p14", "(d) c = 0.01, p14: neutral Gaussian train at low c", sp_i=0, lx=0.41)

fig.suptitle("$\\varepsilon_r$ = 2 (strong coupling): trained on c = 0.01, 0.02, 0.04, 0.06, 0.08 (135 accepted runs);\n"
             "c = 0.05 never trained; V2 from 3 initialisation seeds", x=0.07, ha="left", fontsize=10.5, fontweight="bold")
out = os.path.join(R, "figs", "eps2_full.png")
fig.savefig(out, dpi=200)
print("wrote", out)
