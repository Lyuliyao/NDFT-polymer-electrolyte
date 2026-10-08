"""Channel-resolved W_ab(k_1) of the functional (three seeds) and of the pair-kernel functional at both eps_r and all
six concentrations, and the check Gamma(k_1) against 1 + (n/2) sum_ab W_ab(k_1) (salt density n = n_each)."""
from figstyle import *
from learn import data as D
from learn.models import W_of_k, fine_geometry
from learn.evaluate import S_from_W
from learn.protocol import load_model
out = {}
for eps in ("7.5", "2"):
    SYS = system(eps); sps, _ = D.load_all(root=SYS["root"], concs=CONCS)
    geoms = {c: D.geometry_of(sps[c], int(round(sps[c].L / 0.1))) for c in CONCS}; out[eps] = {}
    for lab, d in [("fun", SYS["fun"][0])] + [("pair", SYS["pair"])]:          # 2026-10-06: the single reported model
        cf, ps = load_model(d); out[eps][lab] = {}
        for c in CONCS:
            g = geoms[c]; gf = fine_geometry(g, 8); k = np.asarray(gf.k); k1 = 2 * np.pi / g.L; i1 = int(np.argmin(abs(k - k1)))
            Wk = np.asarray(W_of_k(ps, cf, gf)); S = S_from_W(Wk, k, g.nbar, g.lB)["NN"] / (2 * g.nbar)
            w = {"++": float(Wk[i1, 0, 0]), "--": float(Wk[i1, 1, 1]), "+-": float(Wk[i1, 0, 1])}
            sw = w["++"] + w["--"] + 2 * w["+-"]
            out[eps][lab][f"{c:g}"] = {**w, "sum": sw, "nbar": float(g.nbar), "Gamma": float(1 / S[i1]), "1+n/2 sumW": float(1 + g.nbar * sw / 2)}
json.dump(out, open(os.path.join(ROOT, "paper/figures/kernel_k1.json"), "w"), indent=1)
for eps in out:
    print("==== eps", eps)
    for lab in out[eps]:
        print(f" {lab:7s}", " | ".join(f"{c}: ++{v['++']:7.1f} --{v['--']:7.1f} +-{v['+-']:6.1f} G {v['Gamma']:5.2f} ({v['1+n/2 sumW']:5.2f})" for c, v in out[eps][lab].items()))
