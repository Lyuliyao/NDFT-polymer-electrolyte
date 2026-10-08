"""Long-wavelength modes in the big boxes: one box mode k = k_1/n (n = 2, 4) below the wavenumbers of the training
box, neutral (p30) or neutral + charged (p31), c = 0.04, both eps_r.  For each run the driven-mode amplitude
a_alpha = 2 <n_alpha exp(-i k z)> from MD, its error from the fix ave/chunk series (500 tau per sample, duplicated
blocks of resumed segments removed, real and imaginary parts each corrected by their integrated autocorrelation
time, as in tools/field_acceptance.py), and the deviation of each model's Euler-Lagrange profile in units of it.
    python figs/bigbox_longmode.py"""
import glob, json, os, sys
import numpy as np
B = "/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_bigbox"; S = "/mnt/gs21/scratch/lyuliyao/salt_in_polymer"
sys.path.insert(0, f"{S}/tools"); sys.path.insert(0, "/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2/code")
from field_acceptance import tau_int
from field_trend import read_ave_chunk
MODELS = {"neural functional": ["NF_s0", "NF_s1", "NF_s2"], "pair closure": ["V1"]}


def series_amp(run, which, k):
    steps, n, z = [], [], None
    for f in sorted(glob.glob(os.path.join(run, f"prof_{which}*.dat"))):
        st, zc, nn = read_ave_chunk(f)
        if len(st): steps.append(st); n.append(nn); z = zc
    steps = np.concatenate(steps); n = np.vstack(n)
    last = {s: i for i, s in enumerate(steps)}; keep = np.array(sorted(last.values()))
    steps, n = steps[keep], n[keep]; n = n[np.argsort(steps)]
    a = 2 * (n * np.exp(-1j * k * z)[None, :]).mean(axis=1)
    var = sum(c.var(ddof=1) * 2 * tau_int(c) / len(c) for c in (a.real, a.imag))
    h = len(a) // 2
    return np.sqrt(var), len(a), abs(a[:h].mean() - a[h:].mean())


out = {}
for e in ("eps75", "eps2"):
    for nbox in (2, 4):
        root = f"{B}/{e}_c0.04_x{nbox}"; st = f"{root}/runs/{e}_c0.04_x{nbox}"
        for tag in ("p30", "p31"):
            run = f"{st}/field_{tag}"
            if not os.path.exists(f"{run}/profiles.npz"):
                continue
            d = np.load(f"{run}/profiles.npz", allow_pickle=True)
            z = d["cation_z"]; lz = len(z) * (z[1] - z[0]); k = 2 * np.pi / lz          # the driven mode m = 1 of the long box
            amd = {sp: 2 * (d[f"{sp}_n"] * np.exp(-1j * k * z)).mean() for sp in ("cation", "anion")}
            row = {}
            for sp, which in (("cation", "cat"), ("anion", "ani")):
                sig, ns, dh = series_amp(run, which, k)
                row[sp] = dict(a_md=abs(amd[sp]), err=sig, rel_err=100 * sig / abs(amd[sp]), samples=ns, halves=dh / sig)
                for lab, ms in MODELS.items():
                    dev, rel = [], []
                    for mname in ms:
                        f = f"{root}/runs/learn/{mname}/predict_c004/predicted.npz"
                        if not os.path.exists(f): continue
                        p = np.load(f); key = f"transfer_c0.04_{tag}"
                        if key not in p.files: continue
                        a = 2 * (p[key][0 if sp == "cation" else 1] * np.exp(-1j * k * z)).mean()
                        dev.append(abs(a - amd[sp]) / sig); rel.append(100 * abs(a - amd[sp]) / abs(amd[sp]))
                    if dev: row[sp][lab] = dict(pull=dev, rel=rel)
            out[f"{e} x{nbox} {tag}"] = row
            print(f"{e} x{nbox} {tag}:", "  ".join(
                f"{sp}: |a| {r['a_md']:.2e} MD err {r['rel_err']:.1f}% (halves {r['halves']:.1f}σ, {r['samples']} samples) | "
                + " ".join(f"{lab} {np.mean(r[lab]['rel']):.1f}% = {np.mean(r[lab]['pull']):.1f}σ" for lab in MODELS if lab in r)
                for sp, r in row.items()), flush=True)
json.dump(out, open(f"{B}/bigbox_longmode.json", "w"), indent=1)
