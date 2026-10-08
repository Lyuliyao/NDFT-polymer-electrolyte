#!/usr/bin/env python3

import argparse
import glob
import json
import os

import numpy as np

LIM = 3.0


def one(run, pot_dir):
    tag = os.path.basename(run).replace("field_", "")
    row = {"tag": tag}
    spec_path = os.path.join(pot_dir, tag + ".json")
    if os.path.exists(spec_path):
        s = json.load(open(spec_path))
        row["family"] = s.get("family", "fourier")
        row["kind"] = s["kind"]
        row["pp"] = f"{s['pp_neutral_kT']:.1f}/{s['pp_charged_kT']:.1f}"
        terms = s.get("neutral", []) + s.get("charged", [])
        amax = max([t["A"] for t in terms], default=1.0)
        row["mdrv"] = max([t["m"] for t in terms if t["A"] >= 1e-3 * amax], default=0)

    npz = os.path.join(run, "profiles.npz")
    if os.path.exists(npz):
        d = np.load(npz, allow_pickle=True)
        for sp in ("cation", "anion"):
            pull = np.abs(d[f"{sp}_ybg_pull"])
            row[f"{sp[:3]}_bins2s"] = float((pull < 2).mean())
            row[f"{sp[:3]}_chi2"] = float((pull**2).mean())
            drv = d[f"{sp}_mode_driven"].astype(bool)
            mp = np.abs(d[f"{sp}_mode_pull"])
            row[f"{sp[:3]}_ybg_mode"] = float(mp[drv].max()) if drv.any() else 0.0
            row[f"{sp[:3]}_contrast"] = float(d[f"{sp}_n"].max() / max(d[f"{sp}_n"].min(), 1e-12))

    acc = os.path.join(run, "acceptance.json")
    if os.path.exists(acc):
        a = json.load(open(acc))
        driven = set()
        if os.path.exists(spec_path):
            s2 = json.load(open(spec_path))
            t2 = s2.get("neutral", []) + s2.get("charged", [])
            am = max([t["A"] for t in t2], default=1.0)
            driven = {t["m"] for t in t2 if t["A"] >= 1e-3 * am}
        for sp in ("cation", "anion"):
            mh = {int(k): v for k, v in a.get(sp, {}).get("mode_half_pull", {}).items()}


            dv = [v for m, v in mh.items() if m in driven]
            uv = {m: v for m, v in mh.items() if m not in driven}
            row[f"{sp[:3]}_half"] = max(dv) if dv else None
            row[f"{sp[:3]}_halfu"] = max(uv.values()) if uv else None
            row[f"{sp[:3]}_halfum"] = max(uv, key=uv.get) if uv else None
            v, e = a.get(f"vcm_{sp}"), a.get(f"vcm_{sp}_err")
            row[f"{sp[:3]}_cur"] = abs(v) / e if (v is not None and e) else None
    return row


def verdict(r):
    need = ("cat_ybg_mode", "cat_half", "cat_cur")
    if any(r.get(k) is None for k in need):
        return "missing"
    bad = []
    if max(r.get("cat_ybg_mode", 0), r.get("ani_ybg_mode", 0)) >= LIM:
        bad.append("YBG")
    if max(r.get("cat_half") or 0, r.get("ani_half") or 0) >= LIM:
        bad.append("halves")
    if max(r.get("cat_halfu") or 0, r.get("ani_halfu") or 0) >= LIM:
        bad.append("slow-modes")
    if max(r.get("cat_cur") or 0, r.get("ani_cur") or 0) >= LIM:
        bad.append("current")
    return "PASS" if not bad else "REVIEW:" + ",".join(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("states", nargs="+")
    ap.add_argument("--csv")
    a = ap.parse_args()
    rows_all = []
    for state in a.states:
        pot = os.path.join(state, "potentials")
        runs = sorted(glob.glob(os.path.join(state, "field_p*")))
        print("=" * 108)
        print(f"{state}   {len(runs)} runs")
        print(f"{'tag':4s} {'family':7s} {'kind':8s} {'pp N/psi':9s} {'m':>3s} | "
              f"{'YBG bins<2s':>12s} {'chi2/bin':>9s} {'YBG mode':>9s} | {'halv drv':>8s} {'halv und':>8s} "
              f"{'current':>8s} | {'contrast':>9s} | verdict")
        for run in runs:
            r = one(run, pot)
            r["state"] = state
            v = verdict(r)
            rows_all.append(dict(r, verdict=v))
            f = lambda k, fmt="{:.2f}": ("--" if r.get(k) is None else fmt.format(r[k]))
            print(f"{r['tag']:4s} {r.get('family','?'):7s} {r.get('kind','?'):8s} "
                  f"{r.get('pp','?'):9s} {r.get('mdrv',0):3d} | "
                  f"{f('cat_bins2s','{:.1%}'):>6s}/{f('ani_bins2s','{:.1%}'):<6s} "
                  f"{f('cat_chi2'):>4s}/{f('ani_chi2'):<4s} "
                  f"{f('cat_ybg_mode','{:.1f}'):>4s}/{f('ani_ybg_mode','{:.1f}'):<4s} | "
                  f"{f('cat_half','{:.1f}'):>3s}/{f('ani_half','{:.1f}'):<4s} "
                  f"{f('cat_halfu','{:.1f}'):>3s}/{f('ani_halfu','{:.1f}'):<4s} "
                  f"{f('cat_cur','{:.1f}'):>3s}/{f('ani_cur','{:.1f}'):<3s} | "
                  f"{f('cat_contrast','{:.1f}'):>4s}/{f('ani_contrast','{:.1f}'):<4s} | {v}")
        ok = sum(1 for r in rows_all if r["state"] == state and r["verdict"] == "PASS")
        miss = sum(1 for r in rows_all if r["state"] == state and r["verdict"] == "missing")
        print(f"  -> {ok}/{len(runs)} PASS, {len(runs) - ok - miss} REVIEW, {miss} not analysed")
    if a.csv:
        import csv
        keys = sorted({k for r in rows_all for k in r})
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader(); w.writerows(rows_all)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
