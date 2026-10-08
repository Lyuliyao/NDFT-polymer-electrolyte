"""Fig. 1: model and data pipeline.  python paper/figures/make_fig1.py
Panels: (a) MD under a static planar field and what is measured, (b) the
functional and the training identity, (c) what the trained functional gives.
Insets use real data (held-out run c = 0.04 p07, the quoted model)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

ROOT = os.environ.get("NDFT_FIGURE_ROOT", "/mnt/gs21/scratch/lyuliyao/salt_in_polymer")
sys.path.insert(0, ROOT)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from figstyle import BEST, E75ROOT, system
from fig1_layout import draw_layout
from learn import data as D
from learn.protocol import load_model
from learn.evaluate import bulk_structure
from learn.models import W_of_k, fine_geometry
from learn.evaluate import S_from_W

MM = 1 / 25.4
fam = [f.name for f in font_manager.fontManager.ttflist]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
    "axes.labelsize": 6, "legend.frameon": False, "legend.fontsize": 5.5,
    "axes.spines.right": False, "axes.spines.top": False, "lines.linewidth": 0.9,
})
C_MD, C_CAT, C_ANI, C_MOD, C_PB = "#0b0b0b", "#5185C0", "#C96144", "#2a78d6", "#8a8984"
INK2, BOX, BOXE = "#52514e", "#f6f5f2", "#b9b7ae"

# ---------------------------------------------------------------- data
sps, runs = D.load_all(root=E75ROOT)
cfg, ps = load_model(os.path.join(ROOT, "runs/learn", BEST))
run = [r for r in runs if r.key == "c0.04/p07"][0]
g = D.geometry_of(sps[0.04], run.n.shape[1]); z = (np.arange(run.n.shape[1]) + 0.5) * g.dz
pred = np.load(os.path.join(ROOT, "runs/learn", BEST, "heldout.npz"))["c0.04_p07"]
sk = sps[0.04].sk; n2 = 2 * float(sk["n_each"])
st = bulk_structure(ps, cfg, g); kk = st["S"]["k"]; sel = (kk > 0) & (kk <= 2.1)
gm = json.load(open(system("7.5")["gamma_md"]))
cs_md = sorted(float(c) for c in gm if float(c) <= 0.08)
g_md = [gm[f"{c:g}"]["shell1"]["Gamma"] for c in cs_md]; e_md = [gm[f"{c:g}"]["shell1"]["Gamma_err"] for c in cs_md]
g_mod = []
for c in cs_md:
    gc = D.geometry_of(sps[c], int(round(sps[c].L / 0.1))); gf = fine_geometry(gc, 8); kf = np.asarray(gf.k)
    S = S_from_W(np.asarray(W_of_k(ps, cfg, gf)), kf, gc.nbar, gc.lB)["NN"] / (2 * gc.nbar)
    g_mod.append(1.0 / S[np.argmin(abs(kf - 2 * np.pi / gc.L))])

# ---------------------------------------------------------------- canvas
plt.rcParams["mathtext.fontset"] = "dejavusans"
W_MM, H_MM = 183, 96
fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
ax0 = fig.add_axes([0, 0, 1, 1]); ax0.set_xlim(0, W_MM); ax0.set_ylim(0, H_MM); ax0.axis("off")

def box(x, y, w, h, text, fs=6, fc=BOX, ec=BOXE, lw=0.6, pad=0.6):
    ax0.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size=1.2", fc=fc, ec=ec, lw=lw, zorder=2))
    ax0.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.3)

def arrow(x0, y0, x1, y1, color=INK2, lw=0.8):
    ax0.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=lw, color=color, zorder=4, shrinkA=1, shrinkB=1))

def panel_label(x, y, letter, title):
    ax0.text(x, y, letter.upper(), fontsize=8, weight="bold", va="top")
    ax0.text(x + 3.4, y - 0.2, title, fontsize=7, va="top", weight="bold")

def header(x, y, text):
    ax0.text(x, y, text, fontsize=6, va="bottom", color="#0b0b0b")

def inset(x_mm, y_mm, w_mm, h_mm):
    return fig.add_axes([x_mm / W_MM, y_mm / H_MM, w_mm / W_MM, h_mm / H_MM])

CW, GAP = 54, 7.5                                   # three equal columns
XA = (4, 4 + CW); XB = (XA[1] + GAP, XA[1] + GAP + CW); XC = (XB[1] + GAP, XB[1] + GAP + CW)
TOP = H_MM - 3
draw_layout(ax0)

# ---------------------------------------------------------------- (a)
axb = inset(XA[0] + 1, 63, 24, 24); axb.set_xlim(-0.5, 25); axb.set_ylim(-0.5, 25); axb.set_aspect("equal"); axb.axis("off")
rng = np.random.default_rng(11)
def worm(n=42, step=0.75, kappa=0.32):
    p = [rng.uniform(3, 21, 2)]; th = rng.uniform(0, 2 * np.pi)
    for _ in range(n - 1):
        th += rng.normal(0, kappa)
        q = p[-1] + step * np.array([np.cos(th), np.sin(th)])
        for d in (0, 1):
            if q[d] < 0.8 or q[d] > 23.7:
                th = np.pi - th if d == 0 else -th
                q = p[-1] + step * np.array([np.cos(th), np.sin(th)])
        p.append(q)
    return np.array(p)
chains = [worm() for _ in range(7)]
for p in chains:
    axb.plot(p[:, 0], p[:, 1], color="#b8b6ae", lw=1.6, solid_capstyle="round", solid_joinstyle="round", zorder=1, alpha=0.95)
mono = np.concatenate(chains)
cat = mono[rng.choice(len(mono), 9, replace=False)] + rng.normal(0, 0.35, (9, 2))
cat = np.vstack([cat, rng.uniform(1.5, 23, (4, 2))])
ani = rng.uniform(1.5, 23, (13, 2))
axb.scatter(cat[:, 0], cat[:, 1], s=13, color=C_CAT, zorder=3, lw=0)
axb.scatter(ani[:, 0], ani[:, 1], s=38, color=C_ANI, zorder=3, lw=0)
for q in cat: axb.text(q[0], q[1], "+", ha="center", va="center", fontsize=3.4, color="white", weight="bold", zorder=4)
for q in ani: axb.text(q[0], q[1], "−", ha="center", va="center", fontsize=4.2, color="white", weight="bold", zorder=4)
axb.add_patch(plt.Rectangle((0, 0), 24.5, 24.5, fill=False, lw=0.6, ec=INK2))
for xx, col, sz, lab in ((0.8, C_CAT, 13, "cation"), (9.6, C_ANI, 38, "anion")):
    axb.scatter([xx], [-2.3], s=sz, color=col, lw=0, clip_on=False, zorder=3)
    axb.text(xx + 1.7, -2.3, lab, ha="left", va="center", fontsize=5.2, color=INK2)
axb.plot([17.2, 19.6], [-2.3, -2.3], color="#b8b6ae", lw=1.6, clip_on=False, solid_capstyle="round")
axb.text(20.3, -2.3, "polymer", ha="left", va="center", fontsize=5.2, color=INK2)
axv = inset(XA[0] + 35, 63, 18, 24)
axv.plot(run.V[0], z, color=C_CAT, lw=0.9); axv.plot(run.V[1], z, color=C_ANI, lw=0.9)
axv.set_ylim(0, g.L); axv.set_xlabel(r"$V_\alpha(z)\,/\,k_BT$", labelpad=1); axv.set_ylabel(r"$z/\sigma$", labelpad=0)
axv.set_yticks([0, 12, 24]); axv.set_xticks([-1, 0, 1])
axn = inset(XA[0] + 10, 34, 43, 20)
axn.plot(z, run.n[0], color=C_CAT, label=r"$n_+$"); axn.plot(z, run.n[1], color=C_ANI, label=r"$n_-$")
axn.fill_between(z, run.n[0] - 2 * run.sig_n[0], run.n[0] + 2 * run.sig_n[0], color=C_CAT, alpha=0.25, lw=0)
axn.set_ylabel(r"$n_\alpha(z)\,/\,\sigma^{-3}$", labelpad=1); axn.set_xlim(0, g.L); axn.set_xticklabels([])
axn.legend(loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.8)
axf = inset(XA[0] + 10, 9, 43, 20)
axf.plot(z, run.f[0], color=C_CAT); axf.plot(z, run.f[1], color=C_ANI)
axf.fill_between(z, run.f[0] - 2 * run.sig_f[0], run.f[0] + 2 * run.sig_f[0], color=C_CAT, alpha=0.25, lw=0)
axf.axhline(0, color=BOXE, lw=0.5); axf.set_xlim(0, g.L)
axf.set_ylabel(r"$f^{\rm int}_\alpha(z)\,/\,\epsilon\sigma^{-4}$", labelpad=1); axf.set_xlabel(r"$z/\sigma$", labelpad=1)

# Panel B and headings are drawn by fig1_layout.draw_layout above.

# ---------------------------------------------------------------- (c)
x = XC[0]; w = CW
axp = inset(x + 9, 63, w - 10, 21)
axp.plot(z, run.n[0], color=C_MD, lw=1.0, label="MD"); axp.plot(z, pred[0], color=C_MOD, lw=0.9, ls="--", label="neural functional")
axp.plot(z, run.n[1], color=C_MD, lw=0.6); axp.plot(z, pred[1], color=C_MOD, lw=0.6, ls="--")
axp.set_xlim(0, g.L); axp.set_ylabel(r"$n_\alpha(z)\,/\,\sigma^{-3}$", labelpad=1); axp.set_xlabel(r"$z/\sigma$", labelpad=0.5); axp.set_yscale("log")
axp.set_yticks([0.02, 0.05]); axp.set_yticklabels(["0.02", "0.05"]); axp.minorticks_off()
axp.legend(loc="lower right", ncol=2, handlelength=1.4, columnspacing=0.8, bbox_to_anchor=(1.0, 1.0), borderaxespad=0)
axs = inset(x + 9, 9, (w - 10) * 0.5 - 1.5, 36)
axs.plot(sk["k"], sk["S_ZZ"] / n2, "o", ms=1.8, color=C_MD, label="MD"); axs.plot(kk[sel], st["S"]["ZZ"][sel] / n2, color=C_MOD, label="neural functional")
axs.set_xlabel(r"$k\sigma$", labelpad=0.5); axs.set_ylabel(r"$S_{ZZ}(k)\,/\,2\bar n$", labelpad=1); axs.set_xticks([0, 1, 2]); axs.set_yticks([0, 0.5, 1, 1.5])
axs.text(0.05, 0.96, "$c$ = 0.04", transform=axs.transAxes, va="top", fontsize=5.5, color=INK2)
axg = inset(x + 9 + (w - 10) * 0.5 + 4.5, 9, (w - 10) * 0.5 - 3, 36)
axg.errorbar(cs_md, g_md, yerr=e_md, fmt="o", ms=1.8, color=C_MD, capsize=1.2, lw=0.6)
axg.plot(cs_md, g_mod, "-", color=C_MOD, lw=0.9)
axg.set_xscale("log"); axg.set_xticks([0.01, 0.02, 0.04, 0.08]); axg.set_xticklabels(["0.01", "0.02", "0.04", "0.08"]); axg.minorticks_off()
axg.set_xlabel("$c$", labelpad=0.5); axg.set_ylabel(r"$\Gamma$", labelpad=1); axg.set_yticks([1, 3, 5, 7, 9])

for ext in ("svg", "pdf", "png"):
    fig.savefig(os.path.join(ROOT, "paper/figures/fig1", f"fig1_schematic.{ext}"), dpi=400 if ext == "png" else None)
print("wrote paper/figures/fig1/fig1_schematic.{svg,pdf,png}")
