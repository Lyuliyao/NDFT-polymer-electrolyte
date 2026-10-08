#!/usr/bin/env python3

import argparse
import json

import numpy as np

M_LONG = [3, 4, 5, 6]
M_SHORT = [8, 12, 16, 20]

M_LONGEST = [1, 2]


PP_MIN, PP_MAX = 0.5, 3.0
GAUSS_TOL = 1e-10


def log_uniform(rng, lo, hi):

    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def gaussian_train(lz, period, width, amp, centre):

    q = int(round(lz / period))
    modes = []
    pref = 2.0 * amp * width * np.sqrt(2.0 * np.pi) / period
    for n in range(1, 200):
        k = 2.0 * np.pi * n / period
        c = pref * np.exp(-0.5 * (k * width) ** 2)
        if abs(c) < GAUSS_TOL * max(abs(amp), 1e-30):
            break
        modes.append(dict(m=int(n * q), A=float(c), k=float(k),
                          phase=float(-k * centre)))
    return modes


def merge_modes(modes):

    acc = {}
    for t in modes:
        z = acc.get(t["m"], 0j) + t["A"] * np.exp(1j * t["phase"])
        acc[t["m"]] = z
    out = []
    for m, z in sorted(acc.items()):
        if abs(z) < 1e-14:
            continue
        out.append(dict(m=int(m), A=float(abs(z)), k=float(2 * np.pi * m / merge_modes.lz),
                        phase=float(np.angle(z))))
    return out


def draw_part(rng, lz, pp_target, n_modes, allowed=None):

    ms = rng.choice(allowed if allowed is not None else M_LONG,
                    size=n_modes, replace=False)
    ks = 2.0 * np.pi * ms / lz
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_modes)
    w = rng.uniform(0.4, 1.0, size=n_modes)
    z = np.linspace(0.0, lz, 200001)
    prof = (w[:, None] * np.cos(ks[:, None] * z[None, :] + phases[:, None])).sum(0)
    pp = prof.max() - prof.min()
    amps = w * (pp_target / pp)
    return [dict(m=int(mm), A=float(aa), k=float(kk), phase=float(pp_))
            for mm, aa, kk, pp_ in zip(ms, amps, ks, phases)]


def rescale(modes, lz, pp_target):

    z = np.linspace(0.0, lz, 200001)
    pp = float(np.ptp(profile(modes, z)))
    f = pp_target / pp if pp > 0 else 1.0
    return [dict(t, A=t["A"] * f) for t in modes]


def profile(modes, z):
    out = np.zeros_like(z)
    for t in modes:
        out += t["A"] * np.cos(t["k"] * z + t["phase"])
    return out


def dprofile(modes, z):
    out = np.zeros_like(z)
    for t in modes:
        out -= t["A"] * t["k"] * np.sin(t["k"] * z + t["phase"])
    return out


def _join(terms):

    if not terms:
        return "0.0"
    out = ""
    for coef, fn in terms:
        out += f"{coef:+.12g}*{fn}"
    return out.lstrip("+")


def lmp_force_expr(neutral, charged, sign):

    terms = [(t["A"] * t["k"], f"sin({t['k']:.12g}*z+{t['phase']:.12g})") for t in neutral]
    terms += [(sign * t["A"] * t["k"], f"sin({t['k']:.12g}*z+{t['phase']:.12g})") for t in charged]
    return _join(terms)


def lmp_energy_expr(neutral, charged, sign):

    terms = [(t["A"], f"cos({t['k']:.12g}*z+{t['phase']:.12g})") for t in neutral]
    terms += [(sign * t["A"], f"cos({t['k']:.12g}*z+{t['phase']:.12g})") for t in charged]
    return _join(terms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lz", type=float, required=True, help="production box L_z")
    ap.add_argument("--temp", type=float, default=1.0, help="T* (sets k_B T)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--kind", choices=["both", "neutral", "charged"], default="both")
    ap.add_argument("--family", choices=["fourier", "mixed", "gauss", "long", "long3"], default="fourier",
                    help="fourier: m in {3,4,5,6}; mixed: adds m in {8,12,16,20}; "
                         "gauss: a periodic Gaussian train (well and barrier); "
                         "long: both of m in {1,2}; long3: m in {1,2,3}")
    ap.add_argument("--gauss-period", type=float, default=None,
                    help="L_z/3 or L_z/4; drawn if omitted")
    ap.add_argument("--gauss-offset", action="store_true",
                    help="charged train whose cation and anion wells sit half a "
                         "period apart")
    ap.add_argument("--pp-neutral", type=float, default=None, help="peak-to-peak of V_N in k_B T")
    ap.add_argument("--pp-charged", type=float, default=None, help="peak-to-peak of psi in k_B T")
    ap.add_argument("--tag", default="pot")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-lmp", required=True)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    kT = a.temp
    merge_modes.lz = a.lz

    def pick(v):
        return log_uniform(rng, PP_MIN, PP_MAX) if v is None else v

    neutral, charged = [], []
    ppn = ppc = 0.0
    gauss = None

    if a.family == "gauss":


        period = a.gauss_period or a.lz / float(rng.choice([3, 4]))
        w_well, w_bar = (log_uniform(rng, 0.5, 2.0) for _ in range(2))
        a_well = -log_uniform(rng, 0.5, 4.0) * kT
        a_bar = log_uniform(rng, 0.5, 4.0) * kT
        centre = float(rng.uniform(0.0, period))
        train = (gaussian_train(a.lz, period, w_well, a_well, centre)
                 + gaussian_train(a.lz, period, w_bar, a_bar, centre + 0.5 * period))
        train = merge_modes(train)
        gauss = dict(period=period, width_well=w_well, width_barrier=w_bar,
                     amp_well_kT=a_well / kT, amp_barrier_kT=a_bar / kT,
                     centre=centre, offset=bool(a.gauss_offset))
        if a.gauss_offset:


            charged = train
            ppc = float(np.ptp(profile(charged, np.linspace(0, a.lz, 20001)))) / kT
        else:
            neutral = train
            ppn = float(np.ptp(profile(neutral, np.linspace(0, a.lz, 20001)))) / kT
    elif a.family in ("long", "long3"):
        allowed = M_LONGEST if a.family == "long" else M_LONGEST + [3]
        if a.kind in ("both", "neutral"):
            ppn = pick(a.pp_neutral)
            neutral = draw_part(rng, a.lz, ppn * kT, len(allowed), allowed=allowed)
        if a.kind in ("both", "charged"):
            ppc = pick(a.pp_charged)
            charged = draw_part(rng, a.lz, ppc * kT, len(allowed), allowed=allowed)
    else:
        extra = M_SHORT if a.family == "mixed" else None
        if a.kind in ("both", "neutral"):
            ppn = pick(a.pp_neutral)
            mods = draw_part(rng, a.lz, ppn * kT, int(rng.integers(2, 4)))
            if extra is not None:

                mods = merge_modes(mods + draw_part(rng, a.lz, 0.5 * ppn * kT,
                                                    int(rng.integers(1, 3)), allowed=extra))
                mods = rescale(mods, a.lz, ppn * kT)
            neutral = mods
        if a.kind in ("both", "charged"):
            ppc = pick(a.pp_charged)
            mods = draw_part(rng, a.lz, ppc * kT, int(rng.integers(2, 4)))
            if extra is not None:
                mods = merge_modes(mods + draw_part(rng, a.lz, 0.5 * ppc * kT,
                                                    int(rng.integers(1, 3)), allowed=extra))
                mods = rescale(mods, a.lz, ppc * kT)
            charged = mods

    spec = dict(tag=a.tag, seed=a.seed, kind=a.kind, family=a.family, lz=a.lz,
                temp=a.temp, pp_neutral_kT=ppn, pp_charged_kT=ppc,
                neutral=neutral, charged=charged)
    if gauss:
        spec["gauss"] = gauss
    with open(a.out_json, "w") as f:
        json.dump(spec, f, indent=2)

    with open(a.out_lmp, "w") as f:
        f.write(f"# external potential '{a.tag}' seed={a.seed} kind={a.kind}\n")
        f.write(f"# L_z={a.lz:.10g}  pp(V_N)={ppn:.3f} kT  pp(psi)={ppc:.3f} kT\n")
        f.write(f"# modes m: V_N {[t['m'] for t in neutral]}  psi {[t['m'] for t in charged]}\n")
        f.write(f'variable fz_cat atom "{lmp_force_expr(neutral, charged, +1)}"\n')
        f.write(f'variable fz_ani atom "{lmp_force_expr(neutral, charged, -1)}"\n')

        f.write(f'variable ez_cat atom "{lmp_energy_expr(neutral, charged, +1)}"\n')
        f.write(f'variable ez_ani atom "{lmp_energy_expr(neutral, charged, -1)}"\n')


    z = np.linspace(0.0, a.lz, 20001)
    vn, ps = profile(neutral, z), profile(charged, z)
    vp, vm = vn + ps, vn - ps
    print(f"[{a.tag}] kind={a.kind} seed={a.seed} L_z={a.lz:.4f}")
    print(f"  V_N : pp={np.ptp(vn):.3f} kT (target {ppn:.3f}) modes m={[t['m'] for t in neutral]}")
    print(f"  psi : pp={np.ptp(ps):.3f} kT (target {ppc:.3f}) modes m={[t['m'] for t in charged]}")
    print(f"  V_+ : pp={np.ptp(vp):.3f} kT   V_- : pp={np.ptp(vm):.3f} kT")
    print(f"  max|F_z| = {max(np.abs(dprofile(neutral,z)+dprofile(charged,z)).max(), np.abs(dprofile(neutral,z)-dprofile(charged,z)).max()):.3f} eps/sigma")
    lam = [2*np.pi/t["k"] for t in neutral+charged]
    if lam:
        print(f"  wavelengths: {' '.join(f'{x:.2f}' for x in sorted(lam))} sigma")


if __name__ == "__main__":
    main()
