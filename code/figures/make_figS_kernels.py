"""SI Appendix figure (formerly Fig. 5): learned kernels in real space. a: effective pair interaction at c = 0.04, W_ab(r) + l_B e e / r;
b: W_--(r) at c = 0.01, 0.04, 0.08 (functional) vs the density-independent pair kernel; c: W_++(r), same."""
from figstyle import *
from learn import data as D
from learn.models import W_of_k, fine_geometry
from learn.protocol import load_model
from learn.interpret import W_r_from_k

SYS = system(eps_arg())          # --eps 2 -> SI Appendix version
sps, runs = D.load_all(root=SYS["root"])
geoms = {c: D.geometry_of(sps[c], int(round(sps[c].L / 0.1))) for c in sorted(sps)}
cfg, ps = load_model(SYS["fun"][0]); cfg1, ps1 = load_model(SYS["pair"])
lB = geoms[0.04].lB; r = np.linspace(0.02, 6.0, 500); E = np.array([1.0, -1.0])
def Wr(cf, pr, c):
    gf = fine_geometry(geoms[c], 8); k = np.asarray(gf.k); Wk = np.asarray(W_of_k(pr, cf, gf))
    return {p: W_r_from_k(k, Wk[:, a, b], r) for p, (a, b) in {"++": (0, 0), "--": (1, 1), "+-": (0, 1)}.items()}
W04 = Wr(cfg, ps, 0.04); W1 = Wr(cfg1, ps1, 0.04)
cols = cramp([0.01, 0.04, 0.08]); pcol = {"++": C_CAT, "--": C_ANI, "+-": "#8281B9"}
H = 50
fig = plt.figure(figsize=(183 * MM, H * MM))
def ax_mm(x, y, w, h): return fig.add_axes([x / 183, y / H, w / 183, h / H])
ax = ax_mm(13, 10, 47, 35)
for p, (a, b) in {"++": (0, 0), "--": (1, 1), "+-": (0, 1)}.items():
    ee = E[a] * E[b]
    ax.plot(r, W04[p] + lB * ee / r, "-", color=pcol[p], lw=1.0, label=f"$W_{{{p}}} + l_B e_+e_{'+' if p=='++' else '-'}/r$".replace("e_+e_+", "e_+e_+").replace("e_+e_-", "e_+e_-"))
    ax.plot(r, lB * ee / r, ":", color=pcol[p], lw=0.7)
ax.axhline(0, color=BAND, lw=0.6); ax.set_ylim(-30 * lB / 7.957, 30 * lB / 7.957); ax.set_xlim(0, 4); ax.axvspan(0, 0.5, color="#f1f0ec", zorder=0)
ax.set_xlabel(r"$r/\sigma$", labelpad=1); ax.set_ylabel(r"effective pair interaction / $k_BT$", labelpad=1); panel(ax, "a", x=-0.26)
ax.legend(loc="upper right", handlelength=1.6)
ax.text(0.98, 0.04, f"$c$ = 0.04, $\\varepsilon_r$ = {SYS['eps']}; dotted: bare Coulomb", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.3, color=INK2)
for k_, (p, lett) in enumerate((("--", "b"), ("++", "c"))):
    ax = ax_mm(74 + k_ * 55, 10, 46, 35)
    for c in (0.01, 0.04, 0.08):
        ax.plot(r, Wr(cfg, ps, c)[p], "-", color=cols[c], lw=1.0, label=f"$c$ = {c:g}")
    ax.plot(r, W1[p], "--", color=C_PAIR, lw=0.9, label="pair closure (any $c$)")
    ax.axhline(0, color=BAND, lw=0.6); ax.set_xlim(0, 4); ax.axvspan(0, 0.5, color="#f1f0ec", zorder=0)
    ax.set_xlabel(r"$r/\sigma$", labelpad=1); ax.set_ylabel(f"$W_{{{p}}}(r)$ / $k_BT$", labelpad=1); panel(ax, lett, x=-0.26)
    if k_ == 0: ax.legend(loc="lower right", handlelength=1.6)
save(fig, "si/kernels" + SYS["suffix"])
