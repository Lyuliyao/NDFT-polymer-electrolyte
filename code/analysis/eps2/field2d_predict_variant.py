"""2D predictions (learn.field2d) of retrained variants, cached next to the paper's cache as <tag>_pred_<label>.npz.
    python figs/field2d_predict_variant.py <label> <suffix>      e.g.  kbK2  _kn1softplus_kbK2   (suffix before {_long}_s{seed})"""
import json, os, sys, numpy as np
B = "/mnt/research/MultiscaleML_group/Liyao"; S = "/mnt/gs21/scratch/lyuliyao/salt_in_polymer"
sys.path.insert(0, f"{B}/salt_in_polymer_eps2/code"); sys.path.insert(0, f"{S}/tools"); os.environ.setdefault("SIP_ROOT", f"{B}/salt_in_polymer_eps2")
import jax; jax.config.update("jax_enable_x64", True)
from learn.protocol import load_model
from learn import field2d as F2
from analyze_profiles2d import pot
LABEL, SUF = sys.argv[1], sys.argv[2]; T = "c001_c002_c004_c006_c008"; OUT = f"{B}/salt_in_polymer_field2d"
SYS = {"eps75": dict(L=24.502285, lB=7.957187947537492, state=f"{S}/field2d/eps75_c0.04", root=f"{B}/salt_in_polymer_eps75", tag="_long"),
       "eps2": dict(L=24.416564, lB=float(np.load(f"{S}/eps2/runs/prod_T1.0_c0.04/zero_field/Sk.npz")["lB"]), state=f"{S}/field2d/eps2_c0.04", root=f"{B}/salt_in_polymer_eps2", tag="")}
for sysname, sy in SYS.items():
    for tag in ("p50", "p51", "p52", "p53"):
        spec = json.load(open(f"{sy['state']}/potentials/{tag}.json")); N = int(round(sy["L"] / 0.1)); g = F2.make_geometry2d(sy["L"], N, 480, sy["lB"])
        md = np.load(f"{sy['state']}/field_{tag}/profiles2d.npz", allow_pickle=True); lo = md["lo"]
        x = (np.arange(N) + 0.5) * sy["L"] / N; X, Y = np.meshgrid(x, x, indexing="ij")
        V = np.stack([pot(spec, 1.0, X + lo[0], Y + lo[1]), pot(spec, -1.0, X + lo[0], Y + lo[1])])
        cache = f"{OUT}/{sysname}/{tag}_pred_{LABEL}.npz"; pred, info = {}, {}
        if os.path.exists(cache):
            c = np.load(cache, allow_pickle=True); pred = {k[2:]: c[k] for k in c.files if k.startswith("n_")}; info = json.loads(str(c["info"]))
        for s in (0, 1, 2):
            lab = f"{LABEL} s{s}"
            if lab in pred: continue
            cfg, ps = load_model(f"{sy['root']}/runs/learn/joint_v2_L1_C4_R128x128_H64_{T}_val3{SUF}{sy['tag']}_s{s}")
            pred[lab], info[lab] = F2.el_solve(ps, cfg, g, V)
            print(f"{sysname} {tag} {lab}: {'ok' if info[lab]['converged'] else 'EL FAIL'} {info[lab]['iters']} it, n/nbar [{pred[lab].min() / g.nbar:.2f}, {pred[lab].max() / g.nbar:.2f}]", flush=True)
            np.savez_compressed(cache, lo=lo, info=json.dumps(info), **{f"n_{k}": v for k, v in pred.items()})
print("done")
