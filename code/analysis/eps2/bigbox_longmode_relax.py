"""Relaxation of the driven long-wavelength mode in the big boxes (p30, p31: one box mode k = k_1/n, n = 2, 4).
The mode relaxes on 1/(k^2 D_s), comparable to the transient, so the production mean of its amplitude is biased
low.  For each run: the complex amplitude a(t) of the m = 1 mode from the fix ave/chunk series (500 tau per
sample, duplicated blocks of resumed segments dropped), projected on its mean phase, fitted with
A [1 - exp(-(t + t_tr) / tau)] over the production (t_tr = transient length, from the first sample step).
Error of A from the fit residuals inflated by their integrated autocorrelation time.  Prints A against the plain
production mean and, where predictions exist, the model amplitudes (NF seeds, pair closure).
    python figs/bigbox_longmode_relax.py"""
import glob, json, os, sys
import numpy as np
from scipy.optimize import curve_fit
S = "/mnt/gs21/scratch/lyuliyao/salt_in_polymer"; B = "/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_bigbox"
sys.path.insert(0, f"{S}/tools")
from field_acceptance import tau_int
from field_trend import read_ave_chunk
DT = 0.005          # tau per step
TTR = {("eps75", 2): 4.25e6 * DT, ("eps75", 4): 1.75e7 * DT, ("eps2", 2): 5.5e6 * DT, ("eps2", 4): 2.15e7 * DT}   # transient, tau (submit_c7.sh)
out = {}
for e in ("eps75", "eps2"):
    for nbox in (2, 4):
        for tag in (("p30", "p31", "p62", "p63") if nbox == 2 else ("p30", "p31")):     # p62/p63 (2026-10-06): the k1/2 mode with aperiodic structure superposed
            run = f"{S}/bigbox/{e}_c0.04_x{nbox}/field_{tag}"
            files = sorted(glob.glob(f"{run}/prof_cat*.dat"))
            if not files:
                continue
            row = {}
            for which in ("cat", "ani"):
                steps, n, z = [], [], None
                for f in sorted(glob.glob(f"{run}/prof_{which}*.dat")):
                    st, zc, nn = read_ave_chunk(f)
                    if len(st): steps.append(st); n.append(nn); z = zc
                steps = np.concatenate(steps); n = np.vstack(n)
                last = {s: i for i, s in enumerate(steps)}; keep = np.array(sorted(last.values()))
                steps, n = steps[keep], n[keep]; o = np.argsort(steps); steps, n = steps[o], n[o]
                L = len(z) * (z[1] - z[0]); k = 2 * np.pi / L
                a = 2 * (n * np.exp(-1j * k * z)[None, :]).mean(axis=1)
                h = len(a) // 2
                ph = np.angle(a[h:].mean()); x = (a * np.exp(-1j * ph)).real
                dstep = int(np.median(np.diff(steps))); t_tr = TTR[(e, nbox)]           # transient length, tau (the step counter restarts at production)
                t = (steps - steps[0] + dstep) * DT                                       # time since production start
                f = lambda t, A, tau: A * (1 - np.exp(-(t + t_tr) / tau))
                (A, tau), cov = curve_fit(f, t, x, p0=(x[h:].mean(), t_tr), bounds=([0, 100.0], [np.inf, 50 * t_tr]))
                r = x - f(t, A, tau); ti = tau_int(r)
                errA = np.sqrt(cov[0, 0] * 2 * ti)                    # fit error, inflated by the residual correlation
                err_mean = x.std(ddof=1) * np.sqrt(2 * tau_int(x) / len(x))
                row[which] = dict(k=k, L=L, t_tr=t_tr, t_prod=float(t[-1]), samples=len(x), A=float(A), errA=float(errA),
                                  tau_fit=float(tau), mean=float(x.mean()), err_mean=float(err_mean),
                                  last_half=float(x[h:].mean()), first_eighth=float(x[:len(x) // 8].mean()))
            c = row["cat"]
            line = (f"{e} x{nbox} {tag}: L {c['L']:.0f}, k {c['k']:.3f}, transient {c['t_tr']:.0f} tau, production {c['t_prod']:.0f} tau ({c['samples']} samples); "
                    f"cation: fit A {c['A']*1e3:.3f}±{c['errA']*1e3:.3f} (tau_fit {c['tau_fit']:.0f} tau), mean {c['mean']*1e3:.3f}±{c['err_mean']*1e3:.3f}, "
                    f"last half {c['last_half']*1e3:.3f}, first eighth {c['first_eighth']*1e3:.3f}  [x1e-3]")
            # models, where predicted
            root = f"{B}/{e}_c0.04_x{nbox}"; pz = np.load(f"{run}/profiles.npz", allow_pickle=True) if os.path.exists(f"{run}/profiles.npz") else None
            if pz is not None:
                zz = pz["cation_z"]; LL = len(zz) * (zz[1] - zz[0]); kk = 2 * np.pi / LL
                for lab, names in (("NF", ["NF_s0", "NF_s1", "NF_s2"]), ("pair", ["V1"]), ("pairKN", ["V1KN"]), ("kb1", ["KB1_s0", "KB1_s1", "KB1_s2"]), ("kb2", ["KB2_s0", "KB2_s1", "KB2_s2"]), ("kb4", ["KB4_s0", "KB4_s1", "KB4_s2"]), ("kn2kb2", ["KN2KB2_s0", "KN2KB2_s1", "KN2KB2_s2"]), ("kbA2", ["KBA2_s0", "KBA2_s1", "KBA2_s2"]), ("kbA4", ["KBA4_s0", "KBA4_s1", "KBA4_s2"]), ("kbK2", ["KBK2_s0", "KBK2_s1", "KBK2_s2"]), ("kbK4", ["KBK4_s0", "KBK4_s1", "KBK4_s2"]), ("pairK", ["V1K"])):
                    amps = []
                    for m in names:
                        fp = f"{root}/runs/learn/{m}/predict_c004/predicted.npz"
                        if os.path.exists(fp) and f"transfer_c0.04_{tag}" in np.load(fp).files:
                            amps.append(abs(2 * (np.load(fp)[f"transfer_c0.04_{tag}"][0] * np.exp(-1j * kk * zz)).mean()))
                    if amps:
                        row[f"model_{lab}_cation"] = amps
                        line += f" | {lab} {np.mean(amps)*1e3:.3f} = {100*(1-np.mean(amps)/c['A']):+.1f}% below A ({(c['A']-np.mean(amps))/c['errA']:.1f}σ)"
            print(line, flush=True)
            out[f"{e} x{nbox} {tag}"] = row
json.dump(out, open(f"{B}/bigbox_longmode_relax.json", "w"), indent=1)
