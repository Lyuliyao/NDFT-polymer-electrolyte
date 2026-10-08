"""Two follow-ups of the comparison of architectures (figs/arch_summary.py), accuracy only:
(1) training-set size: ours, the Sammuller window network and the Cheng CACE functional (window /
    stencil chosen by validation chi2) retrained on nested fractions of the training runs
    (suffix _f025, _f050; fraction 1 = the main runs);
(2) model selection: the same architectures trained for all 4000 steps and scored with the final
    parameters (suffix _<tag>_last) instead of the best-validation checkpoint.
    SIP_ROOT=<root> python figs/arch_extra_summary.py      (cwd = <root>/code)
"""
import json, os, sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

R = os.environ["SIP_ROOT"]; L = f"{R}/runs/learn"
sys.path.insert(0, f"{R}/code")
import learn.data as D
from learn.protocol import load_model
from learn.models import mu_theta, fine_geometry, ESIGN
from learn.geometry import coulomb_vk

T = "c001_c002_c004_c006_c008"
eps2 = "eps2" in os.path.basename(R)
tag = "full" if eps2 else "long"
ours_main = [f"joint_v2_L1_C4_R128x128_H64_{T}_val3_kbK2_{tag}_s{s}" for s in (0, 1, 2)]   # 2026-10-04: spectrum loss (_kbK2)
name = lambda arch, suf, s: f"joint_{arch}_L1_C4_R128x128_H64_{T}_val3{suf}_kbK2_{tag}_s{s}"
done = lambda d: os.path.exists(f"{L}/{d}/predict_c005/metrics.json")
sps, runs = D.load_all()
CS = [0.01, 0.02, 0.04, 0.05, 0.06, 0.08]
geoms = {c: D.geometry_of(sps[c], int(round(sps[c].L / 0.1))) for c in CS}
gm = json.load(open(f"{L}/interpretation/gamma_md.json")) if os.path.exists(f"{L}/interpretation/gamma_md.json") else \
     json.load(open("/mnt/gs21/scratch/lyuliyao/salt_in_polymer/runs/learn/interpretation/gamma_md.json"))
p27 = lambda r: eps2 and abs(r["conc"] - 0.04) < 1e-9 and r["tag"] == "p27"
l2 = lambda r: 100 * 0.5 * (r["profile"]["cation"]["rel_l2"] + r["profile"]["anion"]["rel_l2"])


def gamma_resp(ps, cfg, c):
    g = geoms[c]; gf = fine_geometry(g, 8); N = gf.z.shape[0]
    n0 = jnp.full((2, N), gf.nbar); i0 = N // 2
    f = lambda n: mu_theta(ps, cfg, n, gf)
    K = []
    for b in range(2):
        t = jnp.zeros((2, N)).at[b, i0].set(1.0)
        _, dmu = jax.jvp(f, (n0,), (t,))
        K.append(np.asarray(jnp.fft.rfft(dmu) / jnp.fft.rfft(t[b])[None, :]))
    K = np.stack(K, -1)                                   # (alpha, k, beta)
    k = np.asarray(gf.k); i = int(np.argmin(abs(k - 2 * np.pi / g.L)))
    M = np.eye(2) / g.nbar + coulomb_vk(k[i:i + 1], g.lB)[0] * np.outer(ESIGN, ESIGN) + K[:, i, :]
    return 2 * g.nbar / abs(np.ones(2) @ np.linalg.solve(M, np.ones(2)))


def stats(d):
    m = json.load(open(f"{L}/{d}/metrics.json")); p = json.load(open(f"{L}/{d}/predict_c005/metrics.json"))
    cfg, ps = load_model(f"{L}/{d}")
    ho = [r for r in m["heldout_rows"] if not p27(r)]
    s = dict(ntr=len(m["train_runs"]), ho=np.mean([r["chi2"] for r in ho]), L2=np.mean([l2(r) for r in ho]),
             c5=p["transfer"]["chi2"], c5L2=100 * 0.5 * (p["transfer"]["cation_rel_l2"] + p["transfer"]["anion_rel_l2"]),
             steps=m.get("steps_run", 0))
    q = [r for r in m["heldout_rows"] if p27(r)]
    if q:
        s["p27"] = q[0]["chi2"]
    s["gdev"] = [abs(gamma_resp(ps, cfg, c) - gm[f"{c:g}"]["shell1"]["Gamma"]) / gm[f"{c:g}"]["shell1"]["Gamma_err"] for c in CS]
    return s


def chosen(prefix):
    """The window / stencil the scan used: the variant with a _f050 run present."""
    for arch in prefix:
        if done(name(arch, "_f050", 0)) or done(name(arch, "_f025", 0)):
            return arch
    return prefix[0]


fmt = lambda v: f"{np.mean(v):6.2f}±{np.std(v, ddof=1):.2f}" if len(v) > 1 else f"{np.mean(v):6.2f}"
cols = [("ntr", "train runs"), ("ho", "ho chi2"), ("L2", "ho L2%"), ("c5", "c.05 chi2"), ("c5L2", "c.05 L2%")] + \
       ([("p27", "p27 chi2")] if eps2 else []) + [("gmax", "max|dGamma|/sig"), ("gmean", "mean|dGamma|/sig"), ("steps", "steps")]


def table(title, groups):
    print(title)
    print(f"{'':34s}" + "".join(f"{h:>17s}" for _, h in cols))
    for lab, ds in groups:
        ds = [d for d in ds if done(d)]
        if not ds:
            print(f"{lab:34s}  (not done)"); continue
        ss = [stats(d) for d in ds]
        for s in ss:
            s["gmax"], s["gmean"] = max(s["gdev"]), float(np.mean(s["gdev"]))
        print(f"{lab + f' ({len(ss)})':34s}" + "".join(f"{fmt([s[k] for s in ss if k in s]):>17s}" for k, _ in cols))


print(f"== {os.path.basename(R)}" + ("  (held-out without c=0.04 p27)" if eps2 else ""))
c1 = chosen(["c1win_W30", "c1win_W60"]); ca = chosen(["cace_a5q3", "cace_a5q6"])
groups = []
for frac, suf in (("0.25", "_f025"), ("0.5", "_f050"), ("1", "")):
    groups.append((f"ours, fraction {frac}", ours_main if not suf else [name("v2", suf, s) for s in (0, 1, 2)]))
    groups.append((f"Sammuller {c1[6:]}, fraction {frac}", [name(c1, suf, s) for s in (0, 1, 2)]))
    groups.append((f"Cheng {ca[5:]}, fraction {frac}", [name(ca, suf, s) for s in (0, 1, 2)]))
table("(1) training-set size (best-validation checkpoint)", groups)
lastn = lambda arch, s: f"joint_{arch}_L1_C4_R128x128_H64_{T}_val3_{tag}_last_s{s}"
table("(2) final parameters after 4000 steps (no model selection)",
      [(lab, [lastn(a, s) for s in (0, 1, 2)]) for lab, a in
       (("ours", "v2"), ("Sammuller W30", "c1win_W30"), ("Cheng a5q3", "cace_a5q3"), ("Cheng a5q6", "cace_a5q6"))])
