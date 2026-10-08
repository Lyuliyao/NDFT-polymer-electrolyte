"""Shared style and data helpers for the manuscript figures (print scale, mm)."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

ROOT = os.environ.get("NDFT_FIGURE_ROOT", "/mnt/gs21/scratch/lyuliyao/salt_in_polymer")
sys.path.insert(0, ROOT)
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
CONCS = [0.01, 0.02, 0.04, 0.05, 0.06, 0.08]
def cramp(concs):
    """one-hue sequential colours for concentrations (light -> dark)"""
    t = np.linspace(0.35, 0.95, len(concs))
    return {c: cm.Blues(x) for c, x in zip(concs, t)}
# Model of the paper (2026-10-01): learned radial kernels B_m(r)[1 + g_m(r)], widest 1 sigma, M = 6, softplus; three seeds
# at each eps_r, trained in the research roots. BEST is an absolute path: os.path.join(ROOT, "runs/learn", BEST) returns it unchanged.
E2ROOT = os.environ.get("NDFT_EPS2_ROOT", "/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2")
E75ROOT = os.environ.get("NDFT_EPS75_ROOT", "/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps75")
_T5 = "c001_c002_c004_c006_c008"; _V2N = f"joint_v2_L1_C4_R128x128_H64_{_T5}_val3"
BEST = os.path.join(E75ROOT, "runs/learn", f"{_V2N}_kn1softplus_kbK2_long_s0")   # 2026-10-04: spectrum loss p = 2 (_kbK2)
BEST_GAUSS = "joint_v2_L1_C4_R128x128_H64_c001_c002_c004_c006_c008_val3_long"   # the previous form (Gaussian basis to 2 sigma, SiLU)
V1L = f"joint_v1_L1_C4_R128x128_H64_{_T5}_val3_kn1softplus_kbK2_long_s2"   # 2026-10-08: pair closure with the same learned kernels as the functional, trained like it (seed by validation); before: v1_..._kbK2_long, Gaussian basis, closed form
V1L_E2 = f"joint_v1_L1_C4_R128x128_H64_{_T5}_val3_kn1softplus_kbK2_s0"       # idem at eps_r = 2 (before: v1_..._kbK2_lin)
V2_64 = "joint_v2_L1_C4_c001_c002_c004_c006_c008_val3_long"

def panel(ax, letter, x=-0.28, y=1.06):
    ax.text(x, y, letter.upper(), transform=ax.transAxes, fontsize=8, weight="bold", va="bottom", ha="left")   # PNAS: upper-case panel letters

def save(fig, name):
    d = os.path.join(ROOT, "paper/figures", name); os.makedirs(d, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(os.path.join(d, f"{os.path.basename(name)}.{ext}"), dpi=400 if ext == "png" else None)
    print("wrote", d)

def load_preds(model, runs):
    """EL-predicted profiles saved by the protocol for held-out and predicted runs -> {run.key: (2,N)}"""
    d = os.path.join(ROOT, "runs/learn", model); out = {}
    for f in ("heldout.npz", os.path.join("predict_c005", "predicted.npz")):
        p = os.path.join(d, f)
        if os.path.exists(p):
            z = np.load(p)
            for k in z.files: out[k.replace("transfer_", "")] = z[k]
    return {r.key: out[f"c{r.conc:g}_{r.tag}"] for r in runs if f"c{r.conc:g}_{r.tag}" in out}

def test_runs(runs, ho):
    """20 held-out runs at c = 0.01-0.08 (4 each) plus all PASS runs at the untrained c = 0.05"""
    t = [r for r in runs if r.status == "PASS" and r.conc in CONCS and (r.conc == 0.05 or r.tag in ho.get(r.conc, []))]
    return sorted(t, key=lambda r: (r.conc, r.tag))

def metrics(model):
    return json.load(open(os.path.join(ROOT, "runs/learn", model, "metrics.json")))


# ---- the two systems (eps_r = 7.5 main, eps_r = 2 strong coupling); each trained on its own data
# The paper reports ONE trained model per dielectric constant (user 2026-10-06: no seed statistics): the initialization with
# the lowest validation residual, seed 0 at eps_r = 7.5 and seed 1 at eps_r = 2.  system(e)["fun"] is that single model
# (a one-element list, so that loops over it still work); system(e)["fun_all"] keeps the three initializations.
SEED = {"7.5": 0, "2": 1}


def system(eps="7.5"):
    """root, pair-closure model, the reported functional (one-element list "fun"; "fun_all": all three initializations),
    MD Gamma file and output suffix of one eps_r"""
    if str(eps) == "2":
        L = os.path.join(E2ROOT, "runs/learn")
        return {"eps": "2", "lB": "29.8", "root": E2ROOT, "pair": os.path.join(L, V1L_E2),
                "fun": [os.path.join(L, f"{_V2N}_kn1softplus_kbK2_s{SEED['2']}")],
                "fun_all": [os.path.join(L, f"{_V2N}_kn1softplus_kbK2_s{s}") for s in (0, 1, 2)],
                "gamma_md": gamma_file(L), "suffix": "_eps2"}
    L = os.path.join(E75ROOT, "runs/learn")
    return {"eps": "7.5", "lB": "7.96", "root": E75ROOT, "pair": os.environ.get("NDFT_EPS75_PAIR_MODEL", os.path.join(L, V1L) if os.path.isfile(os.path.join(L, V1L, "config.json")) else os.path.join("/mnt/gs21/scratch/lyuliyao/salt_in_polymer/runs/learn", V1L)),
            "fun": [os.path.join(L, f"{_V2N}_kn1softplus_kbK2_long_s{SEED['7.5']}")],
            "fun_all": [os.path.join(L, f"{_V2N}_kn1softplus_kbK2_long_s{s}") for s in (0, 1, 2)],
            "gamma_md": gamma_file(L), "suffix": ""}
def gamma_file(model_root):
    # 2026-10-07: at eps_r = 7.5 the MD Gamma(k_1) pools nine independent zero-field runs per concentration
    # (tools/zf_pool.py; runs/*/zerofield_rep), interpretation/gamma_md_pooled.json in that model root
    pooled = os.path.join(model_root, "interpretation/gamma_md_pooled.json")
    if os.path.exists(pooled):
        return pooled
    joint = os.path.join(model_root, "interpretation/gamma_md_joint.json")
    return joint if os.path.exists(joint) else os.path.join(model_root, "interpretation/gamma_md.json")
def eps_arg():
    return sys.argv[sys.argv.index("--eps") + 1] if "--eps" in sys.argv else "7.5"
