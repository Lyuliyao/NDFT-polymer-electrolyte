"""Robustness of the Fig. 5 statements: learned kernels W_ab(r) of the functional for three seeds at both eps_r,
and of the pair-kernel functional. Prints values at chosen r and writes kernel_seeds.json."""
from figstyle import *
from learn import data as D
from learn.models import W_of_k, fine_geometry
from learn.protocol import load_model
from learn.interpret import W_r_from_k
r = np.array([0.02, 0.5, 1.0, 1.5, 2.0, 3.0]); rr = np.linspace(0.02, 6.0, 500)
out = {}
for eps in ("7.5", "2"):
    SYS = system(eps); sps, _ = D.load_all(root=SYS["root"], concs=[0.01, 0.04, 0.08])
    geoms = {c: D.geometry_of(sps[c], int(round(sps[c].L / 0.1))) for c in sps}; lB = geoms[0.04].lB
    out[eps] = {"lB": lB}
    for lab, d in [(f"fun_s{i}", d) for i, d in enumerate(SYS["fun"])] + [("pair", SYS["pair"])]:
        cf, ps = load_model(d); out[eps][lab] = {}
        for c in (0.01, 0.04, 0.08):
            gf = fine_geometry(geoms[c], 8); k = np.asarray(gf.k); Wk = np.asarray(W_of_k(ps, cf, gf))
            row = {}
            for p, (a, b) in {"++": (0, 0), "--": (1, 1), "+-": (0, 1)}.items():
                w = W_r_from_k(k, Wk[:, a, b], rr)
                row[p] = {"at_r": [float(np.interp(x, rr, w)) for x in r], "shoulder_max_0.8_2.5": float(w[(rr > 0.8) & (rr < 2.5)].max()),
                          "Wk0": float(Wk[0, a, b])}
            out[eps][lab][f"{c:g}"] = row
json.dump(out, open(os.path.join(ROOT, "paper/figures/kernel_seeds.json"), "w"), indent=1)
for eps in out:
    print(f"==== eps_r = {eps}  (l_B = {out[eps]['lB']:.2f})   r = {list(r)}")
    for p in ("--", "++", "+-"):
        for lab in [k_ for k_ in out[eps] if k_ != "lB"]:
            print(f"W{p} {lab:7s}", "  ".join(f"c={c}: " + " ".join(f"{v:7.1f}" for v in out[eps][lab][c][p]["at_r"]) + f" | sh {out[eps][lab][c][p]['shoulder_max_0.8_2.5']:5.1f}" for c in ("0.01", "0.04", "0.08")))
