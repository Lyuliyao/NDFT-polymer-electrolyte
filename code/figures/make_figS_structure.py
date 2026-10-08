"""SI structure-factor figures (the main-text version is now Fig. 3a, b, make_fig3.py): emergent structure. a: S_ZZ(k)/2n at six concentrations, MD vs the functional;
b: S_NN(k)/2n. The thermodynamic factor Gamma(c) is Fig. 3c (both eps_r); the former panel c was removed 2026-09-29."""
from figstyle import *
from learn import data as D
from learn.models import W_of_k, fine_geometry
from learn.evaluate import bulk_structure, S_from_W, W_interp
from learn.protocol import load_model

SYS = system(eps_arg())          # --eps 2 -> SI Appendix version
sps, runs = D.load_all(root=SYS["root"])
geoms = {c: D.geometry_of(sps[c], int(round(sps[c].L / 0.1))) for c in sorted(sps)}
cfg, ps = load_model(SYS["fun"][0]); cfg1, ps1 = load_model(SYS["pair"])
ALL = "--all" in sys.argv          # SI version with every concentration
# main Fig. 4 (eps_r = 7.5) leaves out c = 0.01, whose MD structure factors are the noisiest (its errors are in the caption);
# the eps_r = 2 SI version keeps c = 0.01, the state point where the functional misses
PLOT_C = CONCS if (SYS["eps"] == "2" or ALL) else [c for c in CONCS if c != 0.01]
cols = cramp(PLOT_C)
gm = json.load(open(SYS["gamma_md"]))
H = 52
fig = plt.figure(figsize=(183 * MM, H * MM))
def ax_mm(x, y, w, h): return fig.add_axes([x / 183, y / H, w / 183, h / H])
axz, axn = ax_mm(14, 10, 74, 36), ax_mm(106, 10, 74, 36)
stats = {}
for c in PLOT_C:
    sk = sps[c].sk; n2 = 2 * float(sk["n_each"]); st = bulk_structure(ps, cfg, geoms[c]); kk = st["S"]["k"]; sel = (kk > 0) & (kk <= 2.1)
    for ax, comp in ((axz, "ZZ"), (axn, "NN")):
        ax.plot(sk["k"], sk[f"S_{comp}"] / n2, "o", ms=1.6, color=cols[c], lw=0, zorder=3)
        ax.plot(kk[sel], st["S"][comp][sel] / n2, "-", color=cols[c], lw=0.9, label=f"{c:g}", zorder=2)
    stats[c] = {}
    for comp in ("ZZ", "NN"):
        md = sk[f"S_{comp}"] / n2; mo = np.interp(sk["k"], kk, st["S"][comp]) / n2
        stats[c][comp] = float(np.sqrt(np.mean((mo - md) ** 2)) / np.sqrt(np.mean(md ** 2)))
axz.set_xlabel(r"$k\sigma$", labelpad=1); axz.set_ylabel(r"$S_{ZZ}(k)\,/\,2\bar n$", labelpad=1); axz.set_ylim(0, 2.1) if SYS["eps"] == "7.5" else axz.set_ylim(bottom=0); panel(axz, "a", x=-0.15)
axn.set_xlabel(r"$k\sigma$", labelpad=1); axn.set_ylabel(r"$S_{NN}(k)\,/\,2\bar n$", labelpad=1); axn.set_ylim(0, 1.05) if SYS["eps"] == "7.5" else axn.set_yscale("log"); panel(axn, "b", x=-0.15)
axz.legend(title="$c$ (points MD, lines neural functional)", loc="upper left", ncol=2, handlelength=1.4, columnspacing=0.8, title_fontsize=5.5)
save(fig, ("si/structure_all" if ALL else "si/structure_75") if SYS["eps"] == "7.5" else "si/structure_eps2")
# rms per concentration for every initialization of the functional and for the pair-kernel functional.
# 2026-10-08: the model S(k) at the MD wavenumbers is computed from W(k) on a box eight times longer (same nbar),
# interpolated by a cubic spline, with the Coulomb term exact at every |k| (as protocol.structure_report, which gives
# Table S7).  The former linear interpolation of S itself between the box-grid points, 2 pi / L apart, added up to 5%
# to the S_ZZ errors (Table S3 disagreed with Table S7).
allst = {}
for lab, d in [(f"fun_s{i}", d) for i, d in enumerate(SYS["fun_all"])] + [("pair", SYS["pair"])]:
    cf_, ps_ = load_model(d); allst[lab] = {}
    for c in CONCS:
        sk = sps[c].sk; n2 = 2 * float(sk["n_each"]); kmd = np.asarray(sk["k"]); g = geoms[c]; gf = fine_geometry(g, 8)
        S = S_from_W(W_interp(np.asarray(gf.k), np.asarray(W_of_k(ps_, cf_, gf)), kmd), kmd, g.nbar, g.lB)
        allst[lab][f"{c:g}"] = {comp: float(np.sqrt(np.mean((S[comp] / n2 - sk[f"S_{comp}"] / n2) ** 2)) / np.sqrt(np.mean((sk[f"S_{comp}"] / n2) ** 2))) for comp in ("ZZ", "NN")}
json.dump(allst, open(os.path.join(ROOT, "paper/figures", "structure_metrics.json" if SYS["eps"] == "7.5" else "si/structure_eps2_metrics.json"), "w"), indent=1)
for lab, d in allst.items(): print(lab, "  ".join(f"{c}: ZZ {v['ZZ']*100:.1f} NN {v['NN']*100:.1f}" for c, v in d.items()))
