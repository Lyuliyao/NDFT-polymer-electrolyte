"""One-label LJ benchmark (the one-component LJTS fluid, r_c 2.5, T 1.5, species merged): the same metrics as
salt_in_polymer_lj/figs/lj_arch_summary.py (two labels), for one species.  Densities 0.1 0.2 0.4 0.6 0.7 trained,
0.5 never trained; Gamma(k_1) = 1/S(k_1); no charges, so no Stillinger-Lovett check and no W_+- asymmetry.
Forms: the paper's functional (learned kernels, kn1softplus), its Gaussian form, Sammuller c1 (+-3, +-6 sigma),
Cheng CACE (R 1.5, 3 sigma), the pair closure; all with the spectrum loss (_kbK2), three seeds; nested fractions
of the training runs (f025, f050) for the accuracy columns.
    SIP_ROOT=<lj1 root> python figs/lj1_arch_summary.py        (cwd = <root>/code)
    -> runs/learn/interpretation/arch_summary.json (full data), arch_frac.json (fractions)"""
import json, os, sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

R = os.environ["SIP_ROOT"]; L = f"{R}/runs/learn"
sys.path.insert(0, f"{R}/code")
import learn.data as D
from learn.protocol import load_model
from learn.models import mu_theta, force_density, W_of_k, fine_geometry
from learn.geometry import coulomb_vk, esign
from learn.evaluate import SPECIES

T = "c01_c02_c04_c06_c07"; tag = "lj1"
V1 = f"v1_{T}_kbK2"
name = lambda arch, f, s: f"joint_{arch}_L1_C4_R128x128_H64_{T}_val3{f}_kbK2_{tag}_s{s}"
FORMS = (("ours (learned kernels)", "v2", "_kn1softplus"), ("ours (Gaussian kernels)", "v2", ""),
         ("Sammuller c1, +-3 sigma", "c1win_W30", ""), ("Sammuller c1, +-6 sigma", "c1win_W60", ""),
         ("Cheng CACE, R 1.5 sigma", "cace_a5q3", ""), ("Cheng CACE, R 3 sigma", "cace_a5q6", ""))
CONFIGS = {"V1 (pair closure)": [V1]}
for lab, arch, kn in FORMS:
    CONFIGS[lab] = [name(arch, kn, s) for s in (0, 1, 2)]
FRACS = {lab: {fr: [name(arch, kn + suf, s) for s in (0, 1, 2)] for fr, suf in (("0.25", "_f025"), ("0.5", "_f050"), ("1", ""))}
         for lab, arch, kn in FORMS if arch in ("v2", "c1win_W30", "cace_a5q3")}
path = lambda d: d if d.startswith("/") else os.path.normpath(f"{L}/{d}")
done = lambda d: os.path.exists(f"{path(d)}/predict_c05/metrics.json")

sps, runs = D.load_all()
fam = {(round(r.conc, 4), r.tag): r.family for r in runs}
CS = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7]
geoms = {c: D.geometry_of(sps[c], int(round(sps[c].L / 0.1))) for c in CS}
passed = [r for r in runs if r.status == "PASS" and any(abs(r.conc - c) < 1e-9 for c in CS)]
gm = json.load(open(f"{L}/interpretation/gamma_md.json"))
labs_of = lambda pm: [s for s in SPECIES if s in pm]
l2 = lambda r: 100 * float(np.mean([r["profile"][s]["rel_l2"] for s in labs_of(r["profile"])]))
tl2 = lambda t: 100 * float(np.mean([t[f"{s}_rel_l2"] for s in SPECIES if f"{s}_rel_l2" in t]))


def K_complex(ps, cfg, g):
    """Complex bulk kernel K_ab(k) (nk, nsp, nsp) of mu_theta at the uniform state."""
    nsp = cfg.n_species; N = g.z.shape[0]; n0 = jnp.full((nsp, N), g.nbar); i0 = N // 2
    f = lambda n: mu_theta(ps, cfg, n, g)
    def one(b):
        t = jnp.zeros((nsp, N)).at[b, i0].set(1.0)
        _, dmu = jax.jvp(f, (n0,), (t,))
        return jnp.fft.rfft(dmu) / jnp.fft.rfft(t[b])[None, :]
    return np.transpose(np.stack([np.asarray(one(b)) for b in range(nsp)], -1), (1, 0, 2))


def gamma_k1(K, k, g):
    """(resp, sym) at k_1: nsp nbar / (e^T S e) with S = (nbar^-1 + v_C e e^T + K)^-1; = 1/S(k_1) per particle."""
    nsp = K.shape[-1]; i = int(np.argmin(abs(k - 2 * np.pi / g.L)))
    ee = np.outer(esign(nsp), esign(nsp)); e = np.ones(nsp)
    M = np.eye(nsp) / g.nbar + coulomb_vk(k[i:i + 1], g.lB)[0] * ee
    Ki = K[i]; Kh = 0.5 * (Ki + Ki.conj().T)
    resp = nsp * g.nbar / abs(e @ np.linalg.solve(M + Ki, e))
    sym = nsp * g.nbar / float(np.real(e @ np.linalg.solve(M + Kh, e)))
    return resp, sym


def Sk_err(K, g, zf):
    """rms relative error of S(k) (Hermitian part of K) against Sk_grid.npz at k = 2 pi m / L (one species: S_pp = S_NN)."""
    z = np.load(os.path.join(zf, "Sk_grid.npz")); nsp = K.shape[-1]
    m = np.asarray(z["m"]).astype(int); ks = np.asarray(z["k"])
    Kh = 0.5 * (K[m] + np.conj(np.swapaxes(K[m], -1, -2)))
    S = np.linalg.inv(np.eye(nsp)[None] / g.nbar + np.real(Kh))
    return 100 * float(np.sqrt(np.mean(((S[:, 0, 0] - z["S_pp"]) / z["S_pp"]) ** 2)))


def structure_checks(ps, cfg, m):
    out = {}; nsp = cfg.n_species
    fd = {c: jax.jit(lambda n, g=g: force_density(ps, cfg, n, g)) for c, g in geoms.items()}
    mt = {c: jax.jit(lambda n, g=g: mu_theta(ps, cfg, n, g)) for c, g in geoms.items()}
    jc = {c: jax.jit(jax.jacfwd(lambda n, g=g: mu_theta(ps, cfg, n, g))) for c, g in geoms.items()}
    noe, refl = [], []
    for r in passed:
        c = round(r.conc, 2); n = jnp.asarray(r.n)
        f = np.asarray(fd[c](n)); noe.append(abs(f.sum()) / np.abs(f).sum())
        mu = np.asarray(mt[c](n)); muR = np.asarray(mt[c](n[:, ::-1]))[:, ::-1]
        refl.append(np.linalg.norm(muR - mu) / np.linalg.norm(mu - mu.mean(axis=1, keepdims=True)))
    out["noether_max"], out["noether_med"] = float(np.max(noe)), float(np.median(noe))
    out["refl_max"], out["refl_med"] = float(np.max(refl)), float(np.median(refl))
    ho = {(round(r.conc, 4), r.tag) for r in passed} & {(round(x["conc"], 4), x["tag"]) for x in m["heldout_rows"]}
    hor = sorted([r for r in passed if (round(r.conc, 4), r.tag) in ho], key=lambda r: (r.conc, r.tag))
    ja = []
    for r in hor:
        n = jnp.asarray(r.n); N = n.shape[1]
        J = np.asarray(jc[round(r.conc, 2)](n)).reshape(nsp * N, nsp * N)
        ja.append(np.linalg.norm(J - J.T) / np.linalg.norm(J))
    out["jac_asym_max"], out["jac_asym_med"] = float(np.max(ja)), float(np.median(ja))
    xg, wg = np.polynomial.legendre.leggauss(16); xg = 0.5 * (xg + 1); wg = 0.5 * wg
    lp = []
    for c in sorted({r.conc for r in hor}):
        rs = [r for r in hor if r.conc == c]
        g = geoms[round(c, 2)]; nb = jnp.full(rs[0].n.shape, g.nbar); mu = mt[round(c, 2)]
        leg = lambda a, b: float(sum(w * jnp.sum(mu(a + s * (b - a)) * (b - a)) for s, w in zip(xg, wg)) * g.A * g.dz)
        for i in range(0, len(rs) - 1, 2):
            n1, n2 = jnp.asarray(rs[i].n), jnp.asarray(rs[i + 1].n)
            I = [leg(nb, n1), leg(n1, n2), leg(n2, nb)]
            lp.append(abs(sum(I)) / sum(abs(x) for x in I))
    out["loop_max"], out["loop_med"] = float(np.max(lp)), float(np.median(lp))
    hm = []
    for c in CS:
        g = geoms[c]; K = K_complex(ps, cfg, g); k = np.asarray(g.k); sel = (k > 0) & (k <= 2.1)
        Ks = K[sel]; hm.append(np.linalg.norm(Ks - np.conj(np.swapaxes(Ks, -1, -2))) / np.linalg.norm(Ks))
    out["W_asym_max"], out["herm_max"], out["SL_max"] = float("nan"), float(max(hm)), float("nan")
    return out


def model_stats(d, checks=True):
    m = json.load(open(f"{path(d)}/metrics.json")); p = json.load(open(f"{path(d)}/predict_c05/metrics.json"))
    cfg, ps = load_model(path(d))
    ho = m["heldout_rows"]; g_ = [r for r in ho if fam.get((round(r["conc"], 4), r["tag"])) == "gauss"]
    s = dict(ho=np.mean([r["chi2"] for r in ho]), ho_med=np.median([r["chi2"] for r in ho]), L2=np.mean([l2(r) for r in ho]), L2g=np.mean([l2(r) for r in g_]),
             c5=p["transfer"]["chi2"], c5L2=tl2(p["transfer"]), el=sum(r["el"]["converged"] for r in ho + p["transfer_rows"]),
             nel=len(ho) + len(p["transfer_rows"]), ntrain=len(m["train_runs"]),
             params=sum(int(np.size(x)) for x in jax.tree_util.tree_leaves(ps)), steps=m.get("steps_run", 0), fit_min=m.get("fit_seconds", 0) / 60)
    gam = {}
    for c in CS:
        gf = fine_geometry(geoms[c], 8); K = K_complex(ps, cfg, gf); gam[c] = gamma_k1(K, np.asarray(gf.k), geoms[c])
        if abs(c - 0.5) < 1e-9:
            s["Sk5"] = Sk_err(K_complex(ps, cfg, geoms[c]), geoms[c], sps[c].zero_field)
    s["gamma"] = gam
    if checks:
        s.update(structure_checks(ps, cfg, m))
    return s


res = {}
for lab, ds in CONFIGS.items():
    ds = [d for d in ds if done(d)]
    if ds:
        res[lab] = [model_stats(d, checks=not lab.startswith("V1")) for d in ds]
        print(f"  {lab}: {len(ds)} models", file=sys.stderr, flush=True)
fmt = lambda v, f=".2f": (f"{np.mean(v):{f}}" if len(v) == 1 else f"{np.mean(v):{f}}±{np.std(v, ddof=1):{f}}")
print(f"== {os.path.basename(R)} (one species)")
cols = [("params", "params", ".0f"), ("ho", "ho chi2", ".2f"), ("ho_med", "ho median", ".2f"), ("L2", "ho L2%", ".2f"), ("L2g", "Gauss L2%", ".2f"),
        ("c5", "rho.5 chi2", ".2f"), ("c5L2", "rho.5 L2%", ".2f"), ("Sk5", "S(k).5 %", ".1f"), ("steps", "steps", ".0f"), ("fit_min", "fit min", ".0f")]
print(f"{'':26s}" + "".join(f"{h:>14s}" for _, h, _ in cols) + f"{'EL conv':>10s}")
for lab, ss in res.items():
    print(f"{lab + f' ({len(ss)})':26s}" + "".join(f"{fmt([s[k] for s in ss if k in s], f):>14s}" if any(k in s for s in ss) else f"{'-':>14s}" for k, _, f in cols)
          + f"{sum(s['el'] for s in ss):>6d}/{sum(s['nel'] for s in ss)}")
print("Gamma(k_1) = 1/S(k_1): full linear response (deviation in MD sigma); [Hermitian part] where they differ by > 1%")
print(f"{'MD':26s}" + "".join(f"{gm[f'{c:g}']['shell1']['Gamma']:9.2f}±{gm[f'{c:g}']['shell1']['Gamma_err']:.2f}      " for c in CS))
for lab, ss in res.items():
    row = f"{lab:26s}"
    for c in CS:
        md, er = gm[f"{c:g}"]["shell1"]["Gamma"], gm[f"{c:g}"]["shell1"]["Gamma_err"]
        r_ = np.mean([s["gamma"][c][0] for s in ss]); y_ = np.mean([s["gamma"][c][1] for s in ss])
        row += f"{r_:8.2f}({(r_ - md) / er:+5.1f})" + (f"[{y_:5.2f}]" if abs(y_ / r_ - 1) > 0.01 else "       ")
    print(row)
print("Structure on the MD profiles (max over runs / state points; median in parentheses)")
sc = [("noether", "Noether force"), ("jac_asym", "Jacobian asym"), ("herm", "K non-Hermitian"), ("refl", "reflection"), ("loop", "loop integral")]
print(f"{'':26s}" + "".join(f"{h:>20s}" for _, h in sc))
for lab, ss in res.items():
    if "noether_max" not in ss[0]:
        continue
    print(f"{lab:26s}" + "".join(f"{max(s[f'{k}_max'] for s in ss):11.1e}" + (f" ({np.median([s[f'{k}_med'] for s in ss]):.0e})" if f"{k}_med" in ss[0] else "         ") for k, _ in sc))
# fractions of the training runs (accuracy only)
frac = {}
for lab, fr in FRACS.items():
    frac[lab] = {}
    for f, ds in fr.items():
        ds = [d for d in ds if done(d)]
        if ds:
            frac[lab][f] = [model_stats(d, checks=False) for d in ds]
print("Training fraction: held-out chi2 (median) / held-out L2% / rho=0.5 chi2 / rho=0.5 L2%  [train runs]")
for lab, fr in frac.items():
    for f, ss in sorted(fr.items(), key=lambda x: float(x[0])):
        print(f"{lab + ', fraction ' + f + f' ({len(ss)})':40s} {np.mean([s['ho'] for s in ss]):8.2f} ({np.mean([s['ho_med'] for s in ss]):.2f}) / {np.mean([s['L2'] for s in ss]):5.2f} / "
              f"{np.mean([s['c5'] for s in ss]):8.2f} / {np.mean([s['c5L2'] for s in ss]):5.2f}   [{ss[0]['ntrain']}]")
os.makedirs(f"{L}/interpretation", exist_ok=True)
conv = lambda ss: [{k: (v if k != "gamma" else {f"{c:g}": list(x) for c, x in v.items()}) for k, v in s.items()} for s in ss]
json.dump({lab: conv(ss) for lab, ss in res.items()}, open(f"{L}/interpretation/arch_summary.json", "w"), indent=1, default=float)
json.dump({lab: {f: conv(ss) for f, ss in fr.items()} for lab, fr in frac.items()}, open(f"{L}/interpretation/arch_frac.json", "w"), indent=1, default=float)
