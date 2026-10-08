"""Profile errors of the two-dimensional tests (2026-10-07, user: report a profile error for every 2D example, as for the
planar profiles; the rms deviation in MD standard errors has no reference).  For every potential (p50-p55), both eps_r,
and the reported neural functional (seed 0 at eps_r = 7.5, seed 1 at eps_r = 2), the pair closure and Poisson-Boltzmann:
  full   ||n_pred - n_MD|| / ||n_MD|| over all bins of the map (0.1 sigma), the planar definition; MD noise
         sqrt(sum s_bin^2) / ||n_MD|| (8-block errors of the map)
  res    the same restricted to the Fourier modes resolved by MD (|a_MD| > 3 standard errors, errors from the block
         series with their autocorrelation time): sqrt(2 sum_res |a_pred - a_MD|^2) / ||n_MD||, with the noise level
         sqrt(2 sum_res s_a^2) / ||n_MD|| of the same sum
averaged over cations and anions (as the planar profile errors).  Reads the MD maps (profiles2d.npz) and the prediction
caches of figs/field2d_compare.py.   python figs/field2d_profile_errors.py  -> salt_in_polymer_field2d/field2d_profile_errors.json"""
import json, os, sys
import numpy as np
B = "/mnt/research/MultiscaleML_group/Liyao"; S = "/mnt/gs21/scratch/lyuliyao/salt_in_polymer"
sys.path.insert(0, f"{S}/tools")
from field_acceptance import tau_int
SEED = {"eps75": 0, "eps2": 1}
TAGS = ("p50", "p51", "p52", "p53", "p54", "p55")


def err_ac(x):
    f = lambda v: v.var(ddof=1) * 2 * tau_int(v) / len(v)
    return np.sqrt(f(x.real) + f(x.imag))


def modes_of(n, mx, my, L):
    N = n.shape[0]; xc = (np.arange(N) + 0.5) * L / N; k1 = 2 * np.pi / L
    Ex = np.exp(-1j * k1 * xc[:, None] * mx[None, :]); Ey = np.exp(-1j * k1 * xc[:, None] * my[None, :])
    return Ex.T @ n @ Ey / N ** 2


out = {}
for sysn in ("eps75", "eps2"):
    for tag in TAGS:
        md = np.load(f"{S}/field2d/{sysn}_c0.04/field_{tag}/profiles2d.npz", allow_pickle=True)
        c = np.load(f"{B}/salt_in_polymer_field2d/{sysn}/{tag}_pred.npz", allow_pickle=True)
        assert np.allclose(c["lo"], md["lo"])
        L = float(md["lz"]); mx, my = md["mode_mx"], md["mode_my"]
        half = (my[None, :] > 0) | ((my[None, :] == 0) & (mx[:, None] > 0))
        models = {"neural functional": c[f"n_NF s{SEED[sysn]}"], "pair closure": c["n_pair closure"], "Poisson-Boltzmann": c["n_PB"]}
        r = {lab: {"full": [], "res": []} for lab in models}; noise = {"full": [], "res": []}; nres = []
        for a, sp in enumerate(("cation", "anion")):
            n, ne = md[f"{sp}_n"], md[f"{sp}_n_err"]; norm = np.linalg.norm(n)
            ser = md[f"{sp}_mode_n"]; am = ser.mean(0)
            er = np.array([[err_ac(ser[:, i, j]) for j in range(len(my))] for i in range(len(mx))])
            res = half & (np.abs(am) > 3 * er); nres.append(int(res.sum()))
            nb = n.mean(); N = n.shape[0]
            # ||n|| in mode units: the a's are normalized by N^2, so ||n||_bins = N * sqrt(nb^2 + 2 sum_half |a|^2) (all modes)
            noise["full"].append(100 * np.sqrt(np.sum(ne ** 2)) / norm)
            noise["res"].append(100 * N * np.sqrt(2 * np.sum(er[res] ** 2)) / norm)
            for lab, p in models.items():
                r[lab]["full"].append(100 * np.linalg.norm(p[a] - n) / norm)
                da = modes_of(p[a], mx, my, L) - am
                r[lab]["res"].append(100 * N * np.sqrt(2 * np.sum(np.abs(da[res]) ** 2)) / norm)
        out[f"{sysn}/{tag}"] = {"modes_resolved": nres, "noise": {k: float(np.mean(v)) for k, v in noise.items()},
                                **{lab: {k: float(np.mean(v)) for k, v in d.items()} for lab, d in r.items()}}
        o = out[f"{sysn}/{tag}"]
        print(f"{sysn} {tag}: resolved modes {nres}  full map: NF {o['neural functional']['full']:.2f} pair {o['pair closure']['full']:.2f} PB {o['Poisson-Boltzmann']['full']:.1f} (noise {o['noise']['full']:.2f})"
              f"  | resolved: NF {o['neural functional']['res']:.2f} pair {o['pair closure']['res']:.2f} PB {o['Poisson-Boltzmann']['res']:.1f} (noise {o['noise']['res']:.2f})")
json.dump(out, open(f"{B}/salt_in_polymer_field2d/field2d_profile_errors.json", "w"), indent=1)
