"""Grid-independence check (2026-10-08, user: the "grid-independent parameters" row of Table 1 had no test).  The reported
functional and the selected baselines are evaluated on the 42 test profiles of each eps_r resampled from the 0.1 sigma MD
bins to 0.05 sigma and ~0.2 sigma grids (band-limited Fourier resampling of the same profile; parameters unchanged), and the
excess chemical potential mu_theta is compared with its value on the 0.1 sigma grid: rms change over both species, relative
to the rms variation of mu_theta over the box, mean and max over the test runs.  The convolutional free energy cannot be
evaluated on another grid: its read-out fixes the number of grid points.  Output: paper/figures/si/grid_resample.json
    python grid_resample.py"""
from figstyle import *
import jax, jax.numpy as jnp
from learn import data as D
from learn.models import mu_theta, cnn_applicable
from learn.protocol import load_model

SEL = json.load(open(os.path.join(ROOT, "paper/figures/single_model_metrics.json")))["selection"]


def resample(x, N2):
    """Band-limited resampling of the periodic rows x (..., N) to N2 points, same origin (the first point of both grids is at
    z = 0 of the box): the Fourier modes up to the smaller Nyquist frequency are kept, so the N -> 2N grid carries the same
    function and the N -> N/2 grid its low-pass filtered version."""
    N = x.shape[-1]; X = np.fft.rfft(x, axis=-1)
    M = min(N, N2) // 2 + 1
    Y = np.zeros(x.shape[:-1] + (N2 // 2 + 1,), complex); Y[..., :M] = X[..., :M]
    if N2 < N and N2 % 2 == 0:
        Y[..., -1] = Y[..., -1].real
    return np.fft.irfft(Y, n=N2, axis=-1) * N2 / N


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
os.makedirs(os.path.join(ROOT, "paper/figures/si"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "paper/figures/si/grid_resample.json"), "w"), indent=1)
