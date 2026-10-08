"""The learned radial kernels of the (c) models: the modulation 1 + g_m(r) of each Gaussian (three seeds),
and the pair kernels u_ab(r) = sum_m c_abm K_m(r) against those of the Gaussian-basis model.
    SIP_ROOT=<eps2 root> python figs/knet_kernels.py   ->  runs/learn/interpretation/knet_kernels.png
"""
import os, sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.environ["SIP_ROOT"]; L = f"{R}/runs/learn"; sys.path.insert(0, f"{R}/code")
from learn.protocol import load_model
from learn.models import mlp
T = "c001_c002_c004_c006_c008"
r = np.linspace(0.02, 6.0, 600)
fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
for s, ls in zip((0, 1, 2), ("-", "--", ":")):
    cfg, ps = load_model(f"{L}/joint_v2_L1_C4_R128x128_H64_{T}_val3_knet_s{s}")
    w = np.asarray(cfg.widths())
    g = np.asarray(mlp(ps["knet"], jnp.asarray(r / cfg.s_max)[:, None]))            # (nr, M)
    B = (2 * np.pi * w[None, :] ** 2) ** -1.5 * np.exp(-0.5 * (r[:, None] / w[None, :]) ** 2)
    for m in range(len(w)):
        msk = r < 3 * w[m]
        ax[0].plot(r[msk], 1 + g[msk, m], ls, color=plt.cm.viridis(m / (len(w) - 1)), lw=1)
    K = B * (1 + g)
    c = cfg.pair_scale * np.asarray(ps["pair"])
    for i, lab in enumerate(("++", "--", "+-")):
        ax[1 + i].plot(r, K @ c[i], ls, color="C3", lw=1.2, label="learned kernels" if s == 0 else None)
    cq, pq = load_model(f"{L}/joint_v2_L1_C4_R128x128_H64_{T}_val3_lin_s{s}")
    wq = np.asarray(cq.widths()); Bq = (2 * np.pi * wq[None, :] ** 2) ** -1.5 * np.exp(-0.5 * (r[:, None] / wq[None, :]) ** 2)
    cqq = cq.pair_scale * np.asarray(pq["pair"])
    for i in range(3):
        ax[1 + i].plot(r, Bq @ cqq[i], ls, color="k", lw=1.0, label="Gaussian basis" if s == 0 else None)
ax[0].axhline(1, color="0.6", lw=0.6); ax[0].set_xlabel("r / sigma"); ax[0].set_title("1 + g_m(r) within 3 widths (colour: width 0.25 -> 2)")
for i, lab in enumerate(("u_++", "u_--", "u_+-")):
    ax[1 + i].axhline(0, color="0.6", lw=0.6); ax[1 + i].set_xlabel("r / sigma"); ax[1 + i].set_title(f"{lab}(r), k_BT sigma^3")
    ax[1 + i].set_xlim(0, 4)
ax[1].legend(fontsize=8)
plt.tight_layout()
os.makedirs(f"{L}/interpretation", exist_ok=True)
plt.savefig(f"{L}/interpretation/knet_kernels.png", dpi=90)
print("wrote", f"{L}/interpretation/knet_kernels.png")
