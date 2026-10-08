from figstyle import *
import jax, jax.numpy as jnp
from scipy.signal import resample as periodic_resample
from learn import data as D
from learn.models import mu_theta, cnn_applicable
from learn.protocol import load_model

SEL = json.load(open(os.path.join(ROOT, "results/figures/single_model_metrics.json")))["selection"]


def resample(x, N2):

    return periodic_resample(x, N2, axis=-1)


out = {}
for e in ("7.5", "2"):
    SYS = system(e); sps, runs = D.load_all(root=SYS["root"]); ho = D.heldout_tags(runs, root=SYS["root"])
    test = [r for r in runs if r.status == "PASS" and ((r.conc in ho and r.tag in ho[r.conc]) or (abs(r.conc - 0.05) < 1e-9 and int(r.tag[1:]) <= 21))]
    models = {"neural functional": SYS["fun"][0], "pair closure": SYS["pair"],
              "one-body network": f"{SYS['root']}/runs/learn/{SEL[f'c1win {e}']['model']}",
              "lattice free energy": f"{SYS['root']}/runs/learn/{SEL[f'cace {e}']['model']}",
              "convolutional free energy": f"{SYS['root']}/runs/learn/{SEL[f'cnn {e}']['model']}"}
    for lab, d in models.items():
        cfg, ps = load_model(d); res = {"model": os.path.basename(d), "n_test": len(test)}
        for fac in (2, 0.5):
            errs = []
            for r in test:
                sp = sps[r.conc]; N = r.n.shape[1]; N2 = int(round(N * fac))
                g = D.geometry_of(sp, N); g2 = D.geometry_of(sp, N2)
                if not (cnn_applicable(cfg, g) and cnn_applicable(cfg, g2)):
                    errs = None; break
                mu = np.asarray(mu_theta(ps, cfg, jnp.asarray(r.n), g))
                mu2 = resample(np.asarray(mu_theta(ps, cfg, jnp.asarray(resample(r.n, N2)), g2)), N)
                errs.append(float(np.linalg.norm(mu2 - mu) / np.linalg.norm(mu - mu.mean(axis=1, keepdims=True))))
            res[f"dz{0.1 / fac:g}"] = None if errs is None else dict(mean=float(np.mean(errs)), median=float(np.median(errs)), max=float(np.max(errs)))
        out[f"{e} {lab}"] = res
        print(e, lab, {k: v for k, v in res.items() if k.startswith("dz")}, flush=True)
os.makedirs(os.path.join(ROOT, "results/figures/si"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results/figures/si/grid_resample.json"), "w"), indent=1)
