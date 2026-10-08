"""Linear-regime response at c = 0.01, eps_r = 2 (runs p40-p43: V_N = A cos(k_m z + phi),
m = 1 or 2, A = 0.03 or 0.05 k_BT, both ions).

MD: the complex amplitude a(t) = 2 <n_N(z, t) exp(-i k_m z)>_z of each profile block (ave/chunk,
100 tau), its mean, and the error of the real and imaginary parts with their own integrated
autocorrelation time (tools/field_acceptance.mode_halves).  Linear response gives
a = -S_NN(k_m) A exp(i phi), so s = -Re(a exp(-i phi)) / A = S_NN(k_m) (per volume) and
Gamma_lin = 2 nbar / s; the quadrature part must vanish.
Models: the Euler-Lagrange profile for the same V, the same amplitude, and the model's own
Hessian value 2 nbar / S_NN(k_m) (W(k) with Coulomb, as figs/gamma_k1.py) for comparison.
    SIP_ROOT=<eps2 root> python figs/linear_gamma.py
"""
import glob, json, os, sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
R = os.environ["SIP_ROOT"]; sys.path.insert(0, f"{R}/code")
sys.path.insert(0, "/mnt/gs21/scratch/lyuliyao/salt_in_polymer/tools")
from field_trend import read_ave_chunk
from field_acceptance import tau_int
import learn.data as D
from learn.protocol import load_model
from learn.evaluate import el_solve, S_from_W
from learn.models import W_of_k, fine_geometry

T = "c001_c002_c004_c006_c008"; L = f"{R}/runs/learn"
MODELS = {"V2 s0": f"joint_v2_L1_C4_R128x128_H64_{T}_val3_full", "V2 s1": f"joint_v2_L1_C4_R128x128_H64_{T}_val3_full_s1",
          "V2 s2": f"joint_v2_L1_C4_R128x128_H64_{T}_val3_full_s2", "V1": f"v1_{T}_full"}
sps, runs = D.load_all()
c = 0.01; sp = sps[c]
gm = json.load(open(f"{L}/interpretation/gamma_md.json"))[f"{c:g}"]


def md_amplitude(run_dir, k, lz):
    a_all = []
    for which in ("cat", "ani"):
        files = sorted(glob.glob(os.path.join(run_dir, f"prof_{which}*.dat")))
        st, z, n = [], None, []
        for f in files:
            s_, zc, nn = read_ave_chunk(f)
            if len(s_):
                st.append(s_); n.append(nn); z = zc
        st = np.concatenate(st); n = np.vstack(n)[np.argsort(st)]
        a_all.append((n * np.exp(-1j * k * z)[None, :]).mean(axis=1) * 2)
    a = a_all[0] + a_all[1]                                     # n_N = n_+ + n_-
    var = lambda x: x.var(ddof=1) * 2 * tau_int(x) / len(x)
    return a.mean(), np.sqrt(var(a.real)), np.sqrt(var(a.imag)), len(a)


def pred_amplitude(n_pred, z, k):
    return 2 * ((n_pred[0] + n_pred[1]) * np.exp(-1j * k * z)).mean()


models = {lab: load_model(f"{L}/{d}") for lab, d in MODELS.items() if os.path.exists(f"{L}/{d}/config.json")}
out = {}
print(f"c = {c}, eps_r = 2, nbar = {sp.n_pairs / sp.L ** 3:.5f}.  Zero-field MD Gamma(k_1) {gm['shell1']['Gamma']:.3f} +- "
      f"{gm['shell1']['Gamma_err']:.3f}, Gamma(k_2) {gm['shell2']['Gamma']:.3f} +- {gm['shell2']['Gamma_err']:.3f}")
for tag in ("p40", "p41", "p42", "p43"):
    r = next((x for x in runs if abs(x.conc - c) < 1e-9 and x.tag == tag), None)
    if r is None:
        print(f"{tag}: no profiles yet"); continue
    t = r.spec["neutral"][0]; m, A, k, phi = t["m"], t["A"], t["k"], t["phase"]
    nbar = sp.n_pairs / sp.L ** 3
    a, er, ei, nblk = md_amplitude(r.path, k, sp.L)
    rot = np.exp(-1j * phi)
    s_md = -(a * rot).real / A                                 # S_NN(k_m), per volume
    q_md = (a * rot).imag / A
    es = np.sqrt((er * np.cos(phi)) ** 2 + (ei * np.sin(phi)) ** 2) / A
    eq = np.sqrt((er * np.sin(phi)) ** 2 + (ei * np.cos(phi)) ** 2) / A
    G, Ge = 2 * nbar / s_md, 2 * nbar / s_md ** 2 * es
    row = dict(m=m, A=A, status=r.status, blocks=nblk, S_over_2n=s_md / (2 * nbar), S_err=es / (2 * nbar),
               quad_over_err=q_md / eq, Gamma_lin=G, Gamma_lin_err=Ge, dn_over_n=abs(a) / (2 * nbar))
    g = D.geometry_of(sp, r.n.shape[1]); z = np.asarray(g.z)
    for lab, (cfg, ps) in models.items():
        n_pred, info = el_solve(ps, cfg, g, r.V)
        ap = pred_amplitude(n_pred, z, k)
        s_p = -(ap * rot).real / A
        gf = fine_geometry(g, 8); kk = np.asarray(gf.k)
        SN = S_from_W(np.asarray(W_of_k(ps, cfg, gf)), kk, g.nbar, g.lB)["NN"]
        row[lab] = dict(Gamma_EL=2 * nbar / s_p, Gamma_hessian=float(2 * nbar / SN[np.argmin(abs(kk - k))]),
                        converged=info["converged"])
    out[tag] = row
    print(f"{tag} m={m} A={A:.2f} kT [{r.status}] {nblk} blocks: dn_N/2n = {row['dn_over_n']:.3f};  MD Gamma_lin {G:.3f} +- {Ge:.3f}"
          f"  (quadrature {row['quad_over_err']:+.1f} sigma)  |  " +
          "  ".join(f"{lab}: EL {row[lab]['Gamma_EL']:.3f} / Hessian {row[lab]['Gamma_hessian']:.3f}" for lab in models))
json.dump(out, open(f"{L}/interpretation/linear_gamma_c001.json", "w"), indent=1, default=float)
