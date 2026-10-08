#!/usr/bin/env python3
"""Pool the zero-field replicas of one state point (zerofield_rep/r*/modes.npz, tools/zf_modes.py) into Gamma(k) and the
small-k structure factors with errors from the spread between independent replicas.
Per replica and k shell (|n|^2 = 1, 2, 3, ...): the frame series of the shell mean of |rho_N(k,t)|^2 / V (and of
|rho_Z|^2 / V), its mean, and its error from blocks inflated by the integrated autocorrelation time (as learn/gamma_md.py).
Pooled: the mean over replicas; error = max(spread of the replica means / sqrt(R), quadrature of the within-replica errors / R).
Gamma(k) = 2 nbar / S_NN(k).  Optionally the original production run (ions.dump.npz) enters as one more replica.
    zf_pool.py --state <state dir> [--orig <zero-field dir>] [--out gamma_rep.json]
Output in the format of gamma_md.json (shell1, shell2, ...: k, Gamma, Gamma_err, S_NN_over_2n, err, tau_int_tau, replicas)."""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zf_modes import kvectors


def tau_int(x, cmax=None):
    x = np.asarray(x, float); x = x - x.mean(); n = len(x); cmax = max(1, min(cmax or n // 4, n))
    f = np.fft.rfft(x, 2 * n); c = np.fft.irfft(f * np.conj(f))[:cmax] / np.arange(n, n - cmax, -1)
    if c[0] <= 0:
        return 0.5
    c /= c[0]; t = 0.5
    for k in range(1, cmax):
        if c[k] <= 0:
            break
        t += c[k]
    return max(t, 0.5)


def series_stat(s, nblock=10):
    """mean, error (max of block and autocorrelation estimates), tau_int (frames) of one series."""
    s = np.asarray(s, float); nt = len(s); ti = tau_int(s)
    bl = np.array([b.mean() for b in np.array_split(s, min(nblock, nt))])
    return float(s.mean()), float(max(bl.std(ddof=1) / np.sqrt(len(bl)), s.std(ddof=1) * np.sqrt(2 * ti / nt))), float(ti)


def shells_from_modes(rho, n2, L, nsp2=True):
    """{n2: (S_NN series, S_ZZ series)} shell means per frame, per volume."""
    V = L ** 3; out = {}
    rn = rho[:, :, 0] + rho[:, :, 1]; rz = rho[:, :, 0] - rho[:, :, 1]
    for m in sorted(set(n2.tolist())):
        sel = n2 == m
        out[m] = ((np.abs(rn[:, sel]) ** 2).mean(1) / V, (np.abs(rz[:, sel]) ** 2).mean(1) / V)
    return out


def load_orig(zf, n2max):
    """The original production run as a replica: modes from ions.dump.npz (frames of ion positions)."""
    z = np.load(os.path.join(zf, "ions.dump.npz")); cols = list(z["cols"]); d = z["data"]
    ix = [cols.index(c) for c in ("xu", "yu", "zu")]; typ = d[0, :, cols.index("type")].astype(int)
    L = float(z["boxes"][0][0, 1] - z["boxes"][0][0, 0]); nvec, n2 = kvectors(n2max); kvec = 2 * np.pi * nvec / L
    rho = np.zeros((d.shape[0], len(n2), 2), complex)
    for t0 in range(0, d.shape[0], 2000):
        ph = np.exp(-1j * np.einsum("tnd,kd->tnk", d[t0:t0 + 2000][:, :, ix], kvec))
        rho[t0:t0 + 2000, :, 0] = ph[:, typ == 2].sum(1); rho[t0:t0 + 2000, :, 1] = ph[:, typ == 3].sum(1)
    return dict(steps=z["steps"], L=L, N=np.array([(typ == 2).sum(), (typ == 3).sum()]), n2=n2, rho=rho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True); ap.add_argument("--orig", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--dt", type=float, default=0.005)
    a = ap.parse_args()
    reps = []
    for f in sorted(glob.glob(os.path.join(a.state, "zerofield_rep", "r*", "modes.npz"))):
        z = np.load(f); reps.append(("r" + f.split("/r")[-1].split("/")[0], dict(steps=z["steps"], L=float(z["L"]), N=z["N"], n2=z["n2"], rho=z["rho"].astype(complex))))
    if a.orig:
        reps.append(("orig", load_orig(a.orig, int(reps[0][1]["n2"].max()) if reps else 9)))
    if not reps:
        sys.exit("no replicas")
    L = reps[0][1]["L"]; N = reps[0][1]["N"]; nbar = N[0] / L ** 3
    out = dict(L=L, nbar=float(nbar), n_replicas=len(reps), replicas=[r[0] for r in reps],
               frames=[int(len(r[1]["steps"])) for r in reps], tau_each=[float(len(r[1]["steps"]) * (r[1]["steps"][1] - r[1]["steps"][0]) * a.dt) for r in reps],
               uncertainty_method="replica_spread_v1")
    per = [shells_from_modes(r[1]["rho"], r[1]["n2"], r[1]["L"]) for r in reps]
    for m in sorted(per[0]):
        k = 2 * np.pi * np.sqrt(m) / L; st = {}
        for lab, j in (("NN", 0), ("ZZ", 1)):
            vals, errs, tis = zip(*[series_stat(p[m][j]) for p in per])
            vals, errs = np.array(vals), np.array(errs); R = len(vals)
            e_between = vals.std(ddof=1) / np.sqrt(R) if R > 1 else float("nan")
            e_within = np.sqrt((errs ** 2).sum()) / R
            st[lab] = dict(mean=float(vals.mean()), err=float(np.nanmax([e_between, e_within])), err_between=float(e_between), err_within=float(e_within),
                           replica_values=vals.tolist(), replica_errs=errs.tolist(),
                           tau_int_tau=float(np.mean(tis) * (reps[0][1]["steps"][1] - reps[0][1]["steps"][0]) * a.dt))
        x = st["NN"]["mean"] / (2 * nbar); ex = st["NN"]["err"] / (2 * nbar)
        out[f"shell{m}"] = dict(k=float(k), n2=int(m), Gamma=float(1 / x), Gamma_err=float(ex / x ** 2), S_NN_over_2n=float(x), err=float(ex),
                                tau_int_tau=st["NN"]["tau_int_tau"], S_NN=st["NN"], S_ZZ=st["ZZ"])
    o = a.out or os.path.join(a.state, "zerofield_rep", "gamma_rep.json")
    json.dump(out, open(o, "w"), indent=1)
    s1 = out["shell1"]
    print(f"{os.path.basename(a.state)}: {len(reps)} replicas x {np.mean(out['tau_each']):.0f} tau; Gamma(k1) = {s1['Gamma']:.3f} +- {s1['Gamma_err']:.3f} "
          f"(between {s1['S_NN']['err_between'] / (2 * nbar) / s1['S_NN_over_2n'] ** 2:.3f}, within {s1['S_NN']['err_within'] / (2 * nbar) / s1['S_NN_over_2n'] ** 2:.3f}); "
          f"replicas: {np.round(1 / (np.array(s1['S_NN']['replica_values']) / (2 * nbar)), 2).tolist()}")


if __name__ == "__main__":
    main()
