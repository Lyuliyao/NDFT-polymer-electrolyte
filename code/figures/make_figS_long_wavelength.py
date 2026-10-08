"""SI Appendix figure (formerly Fig. 6): the response at the longest wavelengths of the box.
a: the long-wavelength run (m = 1, 2 potential) on which the pair-kernel functional is worst (c = 0.02, p18), total ion
density, MD / PB / pair-kernel functional / functional; b, c: 2n/S_NN(k) - 1 at c = 0.01 and c = 0.08, zero-field MD with error bars on the two
lowest shells against the two functionals; d: m = 1 response amplitude on all 30 long-wavelength runs, MD vs
and pair-kernel functional (open symbols: c = 0.05, never trained)."""
from figstyle import *
from learn import data as D
from learn.models import W_of_k, fine_geometry, ModelConfig, init_params
from learn.evaluate import S_from_W, el_solve
from learn.protocol import load_model

SYS = system(eps_arg())          # --eps 2 -> eps_r = 2 version
sps, runs = D.load_all(root=SYS["root"])
geoms = {c: D.geometry_of(sps[c], int(round(sps[c].L / 0.1))) for c in sorted(sps)}
gm = json.load(open(SYS["gamma_md"]))
models = {"pair": load_model(SYS["pair"]), "fun": load_model(SYS["fun"][0])}
pb = (ModelConfig("pb"), init_params(ModelConfig("pb")))
long = [r for r in runs if r.family in ("long", "long3") and r.status == "PASS"]
def amp(y, mm): return 2 * abs(np.fft.rfft(y)[mm]) / len(y)

H = 56
fig = plt.figure(figsize=(183 * MM, H * MM))
def ax_mm(x, y, w, h): return fig.add_axes([x / 183, y / H, w / 183, h / H])

# a: the long-wavelength run on which the pair-kernel functional is worst (relative L2 of n_N)
def relL2(a, b): return float(np.sqrt(np.mean((a - b) ** 2) / np.mean(b ** 2)))
cands = [r for r in long if r.kind in ("neutral", "both")]
err_pair = {r.key: relL2(el_solve(models["pair"][1], models["pair"][0], geoms[r.conc], r.V)[0].sum(0), r.n.sum(0)) for r in cands}
r0 = max(cands, key=lambda r: err_pair[r.key]); g0 = geoms[r0.conc]; z = np.arange(r0.n.shape[1]) * g0.L / r0.n.shape[1]
met = json.load(open(os.path.join(SYS["fun"][0], "metrics.json"))); split0 = "validation" if r0.key in (met.get("val_runs") or []) else ("training" if r0.key in (met.get("train_runs") or []) else "never trained")
print("panel a run", r0.key, split0, "pair-kernel relL2", err_pair[r0.key])
ax = ax_mm(13, 9, 35, 35)
nN = r0.n.sum(0)
# Sum paired species blocks before estimating the uncertainty of total density.
blocks = np.load(os.path.join(r0.path, "profiles.npz"))
if "block_frames_aligned" in blocks and not bool(blocks["block_frames_aligned"]):
    raise ValueError("Total-density uncertainty requires aligned species blocks")
if "cation_n_blocks" in blocks.files:
    bN = blocks["cation_n_blocks"] + blocks["anion_n_blocks"]
    if r0.fold_q != 1:
        bN = D.fold_periodic(bN, r0.fold_q)
    sN = bN.std(axis=0, ddof=1) / np.sqrt(bN.shape[0])
else:                                   # profiles without stored blocks: species errors added in quadrature
    sN = np.sqrt((np.asarray(r0.sig_n) ** 2).sum(0))
ax.fill_between(z, (nN - 2 * sN) / (2 * g0.nbar), (nN + 2 * sN) / (2 * g0.nbar), color=BAND, lw=0, alpha=0.8)
ax.plot(z, nN / (2 * g0.nbar), color=C_MD, lw=1.0, label="MD")
sol = {}
for lab, (cf, pr), col, ls in (("Poisson–Boltzmann", pb, C_PB, ":"), ("pair closure", models["pair"], C_PAIR, "--"), ("neural functional", models["fun"], C_FUN, "-")):
    n_el, info = el_solve(pr, cf, g0, r0.V); sol[lab] = n_el
    ax.plot(z, n_el.sum(0) / (2 * g0.nbar), color=col, lw=0.9, ls=ls, label=lab)
ax.set_xlim(0, g0.L); ax.set_xlabel(r"$z/\sigma$", labelpad=1); ax.set_ylabel(r"$n_N(z)/2\bar n$", labelpad=1); panel(ax, "a", x=-0.3)
lo = min(nN.min() / (2 * g0.nbar), min(v.sum(0).min() for k_, v in sol.items() if k_ != "Poisson–Boltzmann") / (2 * g0.nbar))
hi = max(nN.max() / (2 * g0.nbar), max(v.sum(0).max() for k_, v in sol.items() if k_ != "Poisson–Boltzmann") / (2 * g0.nbar))
ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo)); ax.set_title(f"$c$ = {r0.conc:g}, $m$ = 1, 2 potential", loc="left", pad=2)
fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(13 / 183, 1.0), ncol=4, handlelength=1.8, columnspacing=1.5, borderaxespad=0.3)

# b, c: 2n/S_NN(k) - 1 at c = 0.01 and c = 0.08
def kernel_panel(ax, c, lett, legend=False):
    g = geoms[c]; sk = sps[c].sk; n2 = 2 * float(sk["n_each"])
    ax.plot(sk["k"], n2 / sk["S_NN"] - 1, "o", ms=1.6, color=C_MD, label="MD, zero field")
    for shell, mk in (("shell1", "o"), ("shell2", "^")):
        d = gm[f"{c:g}"][shell]; s, e = d["S_NN_over_2n"], d["err"]
        ax.errorbar([d["k"]], [1 / s - 1], yerr=[[1 / s - 1 / (s + e)], [1 / (s - e) - 1 / s]], fmt=mk, color=C_MD, ms=2.6, capsize=1.5, lw=0.6)
    gf = fine_geometry(g, 8); kk = np.asarray(gf.k); out = {}
    for lab, (cf, pr), col, ls in (("pair closure", models["pair"], C_PAIR, "--"), ("neural functional", models["fun"], C_FUN, "-")):
        S = S_from_W(np.asarray(W_of_k(pr, cf, gf)), kk, g.nbar, g.lB)["NN"]; sel = kk > 0
        ax.plot(kk[sel], 2 * g.nbar / S[sel] - 1, ls, color=col, lw=0.9, label=lab)
        out[lab] = float(np.interp(gm[f"{c:g}"]["shell1"]["k"], kk[sel], 2 * g.nbar / S[sel] - 1))
    ax.axvspan(0, 2 * np.pi * 2.5 / g.L, color="#f1f0ec", zorder=0); ax.set_xlim(0, 2.5)
    ax.set_xlabel(r"$k\sigma$", labelpad=1); ax.set_ylabel(r"$2\bar n/S_{NN}(k) - 1$", labelpad=1); panel(ax, lett, x=-0.24)
    ax.set_title(f"$c$ = {c:g}", loc="left", pad=2); (ax.set_ylim(-0.5, 11) if c == 0.08 else ax.set_ylim(-0.08, 1.15)) if SYS["eps"] == "7.5" else ax.axhline(0, color=BAND, lw=0.6)
    ax.text(2 * np.pi * 1.5 / g.L, 0.5 if c < 0.05 else 0.22, "$m$ = 1, 2", ha="center", va="center", fontsize=5.3, color=INK2, transform=ax.get_xaxis_transform())
    if legend: ax.legend(loc="upper right", handlelength=1.6)
    return out
k1 = {0.01: kernel_panel(ax_mm(60, 9, 33, 35), 0.01, "b"), 0.08: kernel_panel(ax_mm(105, 9, 33, 35), 0.08, "c", legend=True)}

# d: m = 1 response amplitude on the long-wavelength runs
ax = ax_mm(151, 9, 23, 35)
pts = {(m_, s_): [] for m_ in ("fun", "pair") for s_ in ("train", "c005")}
for r in cands:
    gg = geoms[r.conc]; a_md = amp(r.n.sum(0), 1) / (2 * gg.nbar); s_ = "c005" if r.conc == 0.05 else "train"
    for m_ in ("fun", "pair"):
        n_el, _ = el_solve(models[m_][1], models[m_][0], gg, r.V); pts[(m_, s_)].append((a_md, amp(n_el.sum(0), 1) / (2 * gg.nbar)))
style = {"fun": (C_FUN, "o", "neural functional"), "pair": (C_PAIR, "s", "pair closure")}
for (m_, s_), p in pts.items():
    p = np.array(p); col, mk, txt = style[m_]
    ax.plot(p[:, 0], p[:, 1], mk, ms=2.0 if mk == "o" else 1.8, color=col, mfc=col if s_ == "train" else "white", mew=0.6, label=txt if s_ == "train" else None)
ax.plot([], [], "o", ms=2.0, color=INK2, mfc="white", mew=0.6, label="open: $c$ = 0.05, untrained")
lim = max(0.5, 1.1 * max(max(p_[0], p_[1]) for v in pts.values() for p_ in v)); ax.plot([0, lim], [0, lim], "-", color=BAND, lw=0.6, zorder=0); ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
ax.set_xlabel(r"$|n_N(k_1)|/\bar n$, MD", labelpad=1); ax.set_ylabel(r"$|n_N(k_1)|/\bar n$, predicted", labelpad=1); panel(ax, "d", x=-0.45)
ax.legend(loc="lower left", bbox_to_anchor=(-0.05, 1.02), borderaxespad=0, handlelength=1.0, fontsize=5)
save(fig, "si/long_wavelength" + SYS["suffix"])

# numbers for the notes
rep = {"example_run": r0.key, "k1_values": {str(c): {**k1[c], "MD": 1 / gm[f"{c:g}"]["shell1"]["S_NN_over_2n"] - 1,
       "MD_err": gm[f"{c:g}"]["shell1"]["err"] / gm[f"{c:g}"]["shell1"]["S_NN_over_2n"] ** 2} for c in k1}}
rep["panel_a"] = {"run": r0.key, "split": split0, "kind": r0.kind, "m1_amplitude_MD": float(amp(nN, 1) / (2 * g0.nbar))}
for (m_, s_), p in pts.items():
    p = np.array(p)
    rep[f"m1_amplitude_{m_}_{s_}"] = {"n": len(p), "mean_abs_rel_err": float(np.mean(np.abs(p[:, 1] / p[:, 0] - 1))),
                                     "max_abs_rel_err": float(np.max(np.abs(p[:, 1] / p[:, 0] - 1))),
                                     "amplitude_weighted_rel_err": float(np.sum(np.abs(p[:, 1] - p[:, 0])) / np.sum(p[:, 0])),
                                     "slope": float(np.sum(p[:, 0] * p[:, 1]) / np.sum(p[:, 0] ** 2)),
                                     "rel_err": [round(float(x), 3) for x in p[:, 1] / p[:, 0] - 1]}
rep["m1_amplitude_by_conc"] = {}
for c_ in sorted(set(r.conc for r in cands)):
    for m_ in ("fun", "pair"):
        p = np.array([(amp(r.n.sum(0), 1), amp(el_solve(models[m_][1], models[m_][0], geoms[r.conc], r.V)[0].sum(0), 1)) for r in cands if r.conc == c_])
        rep["m1_amplitude_by_conc"].setdefault(str(c_), {})[m_] = float(np.sum(p[:, 1]) / np.sum(p[:, 0]))
rel = {lab: float(np.sqrt(np.mean((sol[lab].sum(0) - nN) ** 2)) / np.sqrt(np.mean(nN ** 2))) for lab in sol}
rep["example_relL2_nN"] = rel
json.dump(rep, open(os.path.join(ROOT, "paper/figures/si/long_wavelength" + SYS["suffix"] + "_metrics.json"), "w"), indent=1)
print(json.dumps(rep, indent=1))
