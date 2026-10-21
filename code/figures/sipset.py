import os, json
from release_paths import E75ROOT, E2ROOT, EPSXROOT

CSET = "01"
T = "c001_c002_c004_c006_c008"
CONCS = [0.01, 0.02, 0.04, 0.05, 0.06, 0.08]
TRAIN_CONCS = [c for c in CONCS if c != 0.05]
CMIN = 0.0
SFX = "_bw01"
LAB = "BW01"
V2 = f"joint_v2_L1_C4_R128x128_H64_{T}_val3"
V1 = f"joint_v1_L1_C4_R128x128_H64_{T}_val3"
PAT = {"ours": f"{V2}_kn1softplus_kbK2_bw_s{{s}}", "plain": f"{V2}_kn1softplus_bw_s{{s}}",
       "f050": f"{V2}_kn1softplus_f050_kbK2_bw_s{{s}}", "f025": f"{V2}_kn1softplus_f025_kbK2_bw_s{{s}}",
       "pair": f"{V1}_kn1softplus_kbK2_bw_s{{s}}"}
EPSX_BASE = f"joint_v2_L1_C4_R128x128_H64_{T}_"
EPSX_SUF = "lbpoly_val3_kn1softplus_kbK2_bw"
FINAL_SRC = os.path.join(EPSXROOT, "runs/learn", f"{EPSX_BASE}E75-5-4-3-2_{EPSX_SUF}_s0")
LOO = {"5": f"{EPSX_BASE}E75-4-3-2_T5_{EPSX_SUF}_s0", "4": f"{EPSX_BASE}E75-5-3-2_T4_{EPSX_SUF}_s0",
       "3": f"{EPSX_BASE}E75-5-4-2_T3_{EPSX_SUF}_s0"}
VIEW = "FINAL_bwX_01"


def by_val(L, fmt, seeds=(0, 1, 2)):

    c = [fmt.format(s=s) for s in seeds if os.path.isfile(os.path.join(L, fmt.format(s=s), "metrics.json"))]
    return min(c, key=lambda n: json.load(open(os.path.join(L, n, "metrics.json")))["val_chi2"]) if c else fmt.format(s=seeds[0])


def root_of(eps):
    return E75ROOT if str(eps) == "7.5" else E2ROOT
