"""Gamma(k_1) = 2n/S_NN(k_1) and the stability check (smallest eigenvalue of S^-1 on the 8x grid) for the eps_r = 7.5
ablations trained on the full data set (long-wavelength runs included); SI Appendix, section S7, Table S7."""
from figstyle import *
from learn import data as D
from learn.models import W_of_k, fine_geometry
from learn.evaluate import S_from_W
from learn.protocol import load_model
T5 = "c001_c002_c004_c006_c008"
MODELS = {"readout 128x128 (quoted)": BEST, "readout 64x64": f"joint_v2_L1_C4_{T5}_val3_long", "two smoothing layers": f"joint_v2_L2_C4_{T5}_val3_long",
          "without pair term": f"joint_v3_L1_C4_{T5}_val3_long", "mode weighting": f"joint_v2_L1_C4_{T5}_val3_longk"}
sps, _ = D.load_all(concs=CONCS); geoms = {c: D.geometry_of(sps[c], int(round(sps[c].L / 0.1))) for c in CONCS}
out = {}
for lab, m in MODELS.items():
    cf, ps = load_model(os.path.join(ROOT, "runs/learn", m)); out[lab] = {"model": m}
    for c in CONCS:
        g = geoms[c]; gf = fine_geometry(g, 8); k = np.asarray(gf.k); i1 = int(np.argmin(abs(k - 2 * np.pi / g.L)))
        S = S_from_W(np.asarray(W_of_k(ps, cf, gf)), k, g.nbar, g.lB)["NN"] / (2 * g.nbar)
        out[lab][f"{c:g}"] = round(float(1 / S[i1]), 3)
    print(lab, out[lab])
json.dump(out, open(os.path.join(ROOT, "paper/figures/si/ablation_gamma.json"), "w"), indent=1)
