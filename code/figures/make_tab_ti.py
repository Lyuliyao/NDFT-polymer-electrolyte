"""SI table of the free energy of switching on a field (si/ti_free_energy.json, ti_free_energy.py --md): MD by thermodynamic
integration against the closed-form free energies of the functionals, k_B T per ion pair, c = 0.05 (untrained), both eps_r.
    python make_tab_ti.py  -> prints the rows and writes si/ti_rows.tex"""
import json, os, numpy as np
d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "si/ti_free_energy.json")))
NAME = {"p01": "Fourier, neutral + charged", "p07": "Fourier, charged", "p11": "mixed, neutral + charged", "p13": "Gaussian wells, neutral"}
rows = []
for e in ("7.5", "2"):
    for i, tag in enumerate(("p01", "p07", "p11", "p13")):
        r = d[f"{e}/{tag}"]; md = r.get("md_dF")
        fun = [x["dF_closed"] for x in r["functional"]]; pc = r["pair closure"][0]["dF_closed"]; pb = r["PB"][0]["dF_closed"]
        ob = [x["dF_TI"] for x in r.get("one-body network", [])]; ob2 = [x.get("dF_path2") for x in r.get("one-body network", [])]
        dev = lambda v: f"{100 * (v / md[0] - 1):+.1f}" if md else "--"
        cell_ob = (f"{np.mean(ob):.3f} ({dev(np.mean(ob))}\\%)" if ob else "--") + (f"; {np.mean(ob2):.3f}" if ob and ob2[0] is not None else "")
        rows.append(f"{e if i == 0 else ''} & {NAME[tag]} & " + (f"${md[0]:.3f}\\pm{md[1]:.3f}$" if md else "--") +
                    f" & {np.mean(fun):.3f} ({dev(np.mean(fun))}\\%) & {pc:.3f} ({dev(pc)}\\%) & {pb:.3f} ({dev(pb)}\\%) & {cell_ob}\\\\")
        print(rows[-1])
        print(f"      functional seeds {[round(v, 4) for v in fun]}  U1 MD {r['md_U1'][0]:.4f}+-{r['md_U1'][1]:.4f}  fun U1 {np.mean([x['U_lambda']['1.0000'] for x in r['functional']]):.4f}"
              + (f"  MD U(lambda) {[round(v, 4) for v in r['md_U_lambda'].values()]}" if "md_U_lambda" in r else ""))
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "si/ti_rows.tex"), "w").write("\n".join(rows) + "\n")
