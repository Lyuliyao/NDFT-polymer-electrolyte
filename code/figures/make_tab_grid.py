"""SI table S4.5 (grid refinement) from grid_compare.json: held-out chi2 and profile error of the models trained on 0.1 sigma
and on 0.05 sigma bins, each evaluated on both grids (mean +- s.d. over three seeds), and Gamma(k_1) at c = 0.04 (range over seeds).
    python make_tab_grid.py  -> si/si_tab_grid.tex (2026-10-04: spectrum-loss models)"""
from figstyle import *
G = json.load(open("/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2/runs/learn/interpretation/grid_compare.json"))
L = os.path.join(ROOT, "paper/output/latex/si/si_tab_grid.tex")
pm = lambda v: f"${np.mean(v):.3f}\\pm{np.std(v, ddof=1):.3f}$"; pm2 = lambda v: f"${np.mean(v):.2f}\\pm{np.std(v, ddof=1):.2f}$"
rows = []
for e in ("7.5", "2"):
    seeds = [G[f"{e}/s{s}"] for s in range(3)]
    for trained, key in (("0.1", "coarse"), ("0.05", "fine")):
        # evaluations of the model trained on `key` bins: on 0.1 sigma bins and on 0.05 sigma bins
        ev01 = [s[key] if key == "coarse" else s["fine_on_coarse"] for s in seeds]
        ev005 = [s["coarse_on_fine"] if key == "coarse" else s[key] for s in seeds]
        gam = [s["Gamma"]["0.04"][0 if key == "coarse" else 1] for s in seeds]      # [coarse model, fine model, MD, MD err]
        rows.append(f"{e if trained == '0.1' else ''} & {trained} & {pm([x['ho_chi2'] for x in ev01])} & {pm([x['ho_chi2'] for x in ev005])} & {pm2([x['ho_L2'] for x in ev01])} & {pm2([x['ho_L2'] for x in ev005])} & {min(gam):.2f}--{max(gam):.2f}\\\\")
tex = open(L).read()
import re
m = re.search(r"(\\midrule\n)(.*?)(\\bottomrule)", tex, re.S); tex = tex[:m.start(2)] + "\n".join(rows) + "\n" + tex[m.start(3):]
open(L, "w").write(tex); print("\n".join(rows))
