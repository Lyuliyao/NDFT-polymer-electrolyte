import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from release_paths import REPO, GROUP, E75ROOT, E2ROOT, MAIN, FIGURES
import sipset as SS
ROOT = str(REPO)
FIGURES.mkdir(parents=True, exist_ok=True)
(FIGURES / "si").mkdir(exist_ok=True)
os.environ.pop("SIP_EPS_ROOTS", None)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
    "axes.labelsize": 6, "legend.frameon": False, "legend.fontsize": 5.5, "axes.titlesize": 6.5,
    "axes.spines.right": False, "axes.spines.top": False, "lines.linewidth": 0.9,
})
C_MD, C_FUN, C_PAIR, C_PB = "#0b0b0b", "#2a78d6", "#eb6834", "#8a8984"
C_CAT, C_ANI, INK2, BAND = "#5185C0", "#C96144", "#52514e", "#d8d7d2"
CONCS = SS.CONCS
def cramp(concs):

    t = np.linspace(0.35, 0.95, len(concs))
    return {c: cm.Blues(x) for c, x in zip(concs, t)}
os.environ.setdefault("SIP_ROOT", E75ROOT)
_T5 = SS.T; _V2N = SS.V2
def by_val(L, fmt, seeds=(0, 1, 2)):

    c = [fmt.format(s) for s in seeds if os.path.isfile(os.path.join(L, fmt.format(s), "metrics.json"))]
    return min(c, key=lambda n: json.load(open(os.path.join(L, n, "metrics.json")))["val_chi2"]) if c else fmt.format(seeds[0])
FINAL = SS.VIEW
BEST = os.path.join(E75ROOT, "runs/learn", FINAL)
BEST_KBK2 = os.path.join(E75ROOT, "runs/learn", by_val(os.path.join(E75ROOT, "runs/learn"), SS.PAT["ours"].replace("{s}", "{}")))
BEST_GAUSS = "joint_v2_L1_C4_R128x128_H64_c001_c002_c004_c006_c008_val3_long"
V1L = by_val(os.path.join(E75ROOT, "runs/learn"), SS.PAT["pair"].replace("{s}", "{}"))
V1L_E2 = by_val(os.path.join(E2ROOT, "runs/learn"), SS.PAT["pair"].replace("{s}", "{}"))
V2_64 = "joint_v2_L1_C4_c001_c002_c004_c006_c008_val3_long"

def panel(ax, letter, x=-0.28, y=1.06):
    ax.text(x, y, letter.upper(), transform=ax.transAxes, fontsize=8, weight="bold", va="bottom", ha="left")

def save(fig, name):
    d = os.path.join(ROOT, "results/figures", name); os.makedirs(d, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(os.path.join(d, f"{os.path.basename(name)}.{ext}"), dpi=400 if ext == "png" else None)
    print("wrote", d)

def load_preds(model, runs):

    d = os.path.join(MAIN, "runs/learn", model); out = {}
    for f in ("heldout.npz", os.path.join("predict_c005", "predicted.npz")):
        p = os.path.join(d, f)
        if os.path.exists(p):
            z = np.load(p)
            for k in z.files: out[k.replace("transfer_", "")] = z[k]
    return {r.key: out[f"c{r.conc:g}_{r.tag}"] for r in runs if f"c{r.conc:g}_{r.tag}" in out}

def test_runs(runs, ho):

    t = [r for r in runs if r.status == "PASS" and r.conc in CONCS and (r.conc == 0.05 or r.tag in ho.get(r.conc, []))]
    return sorted(t, key=lambda r: (r.conc, r.tag))

def metrics(model):
    return json.load(open(os.path.join(MAIN, "runs/learn", model, "metrics.json")))


SEED = {"7.5": 0, "2": 1}


def system(eps="7.5"):

    if str(eps) == "2":
        L = os.path.join(E2ROOT, "runs/learn")
        return {"eps": "2", "lB": "29.8", "root": E2ROOT, "pair": os.path.join(L, V1L_E2),
                "fun": [os.path.join(L, FINAL)],
                "fun_all": [os.path.join(L, FINAL)],
                "fun_fixed": [os.path.join(L, by_val(L, SS.PAT["ours"].replace("{s}", "{}")))],
                "gamma_md": gamma_file(L), "suffix": "_eps2"}
    L = os.path.join(E75ROOT, "runs/learn")
    return {"eps": "7.5", "lB": "7.96", "root": E75ROOT, "pair": os.environ.get("NDFT_EPS75_PAIR_MODEL", os.path.join(L, V1L)),
            "fun": [os.path.join(L, FINAL)],
            "fun_all": [os.path.join(L, FINAL)],
            "fun_fixed": [os.path.join(L, by_val(L, SS.PAT["ours"].replace("{s}", "{}")))],
            "gamma_md": gamma_file(L), "suffix": ""}
def gamma_file(model_root):
    pooled = os.path.join(model_root, "interpretation/gamma_md_pooled.json")
    if os.path.exists(pooled):
        return pooled
    joint = os.path.join(model_root, "interpretation/gamma_md_joint.json")
    return joint if os.path.exists(joint) else os.path.join(model_root, "interpretation/gamma_md.json")
def eps_arg():
    return sys.argv[sys.argv.index("--eps") + 1] if "--eps" in sys.argv else "7.5"
