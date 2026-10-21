from figstyle import *
import figstyle
from matplotlib.lines import Line2D
from release_paths import EPSXROOT as X, EPS_ROOTS
import sipset as SS
BASE = SS.EPSX_BASE
FINALK = BASE + "E75-5-4-3-2_lbpoly_val3_kn1softplus_kbK2_bw_s0"
LOO = {"5": BASE + "E75-4-3-2_T5_lbpoly_val3_kn1softplus_kbK2_bw_s0", "4": BASE + "E75-5-3-2_T4_lbpoly_val3_kn1softplus_kbK2_bw_s0",
       "3": BASE + "E75-5-4-2_T3_lbpoly_val3_kn1softplus_kbK2_bw_s0"}
INPUTK = BASE + "E75-5-3-2_T4_lbinput_val3_kn1softplus_kbK2_bw_s0"
EPS = ["7.5", "5", "4", "3", "2"]; LB = {"7.5": 7.957187947537492, "5": 11.935781921306239, "4": 14.9197274016328, "3": 19.892969868843736, "2": 29.8394548032656}
COL = {e: cm.viridis(x) for e, x in zip(EPS, np.linspace(0.08, 0.88, 5))}
CC = SS.CONCS
GMD = {}
for e in EPS:
    f = {"7.5": figstyle.gamma_file(f"{E75ROOT}/runs/learn"), "2": figstyle.gamma_file(f"{E2ROOT}/runs/learn")}.get(e, f"{EPS_ROOTS[e]}/runs/learn/interpretation/gamma_md.json")
    g = json.load(open(f)); GMD[e] = {float(c): (g[c]["shell1"]["Gamma"], g[c]["shell1"]["Gamma_err"]) for c in g if SS.CMIN <= float(c) <= 0.085}
GM = json.load(open(f"{X}/gamma_k1_models{SS.SFX}_s0.json"))
def gmod(key, e, c): return GM[key].get(f"{e}/{c:g}")
EN = json.load(open(f"{X}/lb_energy_test2{SS.SFX}.json"))
CH = json.load(open(f"{X}/channel_epsx{SS.SFX}.json"))[FINALK]["states"]
W, H = 183, 104
fig = plt.figure(figsize=(W * MM, H * MM))
def ax_mm(x, y, w, h): return fig.add_axes([x / W, y / H, w / W, h / H])
axA, axB, axC = ax_mm(12, 61, 44, 36), ax_mm(74, 61, 44, 36), ax_mm(136, 61, 44, 36)
axD, axE, axF = ax_mm(12, 10, 44, 36), ax_mm(74, 10, 44, 36), ax_mm(136, 10, 44, 36)
M = {"stats": {}}
dev_loo, dev_all = [], []
for e in EPS:
    cs = sorted(GMD[e]); md = np.array([GMD[e][c][0] for c in cs]); er = np.array([GMD[e][c][1] for c in cs])
    axA.errorbar(cs, md, yerr=er, fmt="o", ms=2.4, color=COL[e], capsize=1.3, lw=0.6, zorder=5, label=f"$\\varepsilon_r={e}$")
    key = LOO.get(e, FINALK); ls = "-" if e in LOO else "-."
    cm_ = [c for c in cs if gmod(key, e, c) is not None]; ym = [gmod(key, e, c) for c in cm_]
    axA.plot(cm_, ym, ls, color=COL[e], lw=0.9, zorder=4)
    for c, y in zip(cm_, ym):
        d = (y - GMD[e][c][0]) / GMD[e][c][1]
        (dev_loo if e in LOO else dev_all).append(d)
        M["stats"][f"Gamma {e}/{c:g}"] = dict(MD=GMD[e][c][0], err=GMD[e][c][1], model=y, dev=d, model_key=key)
M["gamma_rms_loo_sigma"] = float(np.sqrt(np.mean(np.square(dev_loo)))); M["gamma_max_loo_sigma"] = float(np.max(np.abs(dev_loo))); M["gamma_n_loo"] = len(dev_loo)
M["gamma_rms_ends_sigma"] = float(np.sqrt(np.mean(np.square(dev_all)))); M["gamma_max_ends_sigma"] = float(np.max(np.abs(dev_all)))
axA.set_xscale("log"); axA.set_xticks(([0.01] if SS.CSET == "01" else []) + [0.02, 0.04, 0.08]); axA.set_xticklabels((["0.01"] if SS.CSET == "01" else []) + ["0.02", "0.04", "0.08"]); axA.minorticks_off()
axA.axhline(1, color=INK2, lw=0.5, ls=":"); axA.set_xlabel("$c$", labelpad=1); axA.set_ylabel(r"$\Gamma(k_1)$", labelpad=1)
legA = axA.legend(loc="lower right", ncol=2, fontsize=5, handlelength=1.0, handletextpad=0.3, labelspacing=0.15, columnspacing=0.6, borderaxespad=0.1); axA.add_artist(legA)
LSTY = [Line2D([], [], color=INK2, lw=0.9, ls="-", label="left-out functional"), Line2D([], [], color=INK2, lw=0.9, ls="-.", label="five-permittivity functional")]
axA.legend(handles=LSTY, loc="upper left", fontsize=5, handlelength=1.6, handletextpad=0.5, labelspacing=0.2, borderaxespad=0.2); panel(axA, "a", x=-0.24, y=1.04)
from scipy.interpolate import CubicSpline
MODE = json.load(open(f"{X}/lb_energy_models{SS.SFX}.json"))
FILMK = BASE + "E75-5-3-2_T4_lbfilm_val3_kn1softplus_kbK2_bw_s0"
def d2fd(x, y):
    return np.array([2 * ((x[i] - x[i-1]) * y[i+1] - (x[i+1] - x[i-1]) * y[i] + (x[i+1] - x[i]) * y[i-1]) / ((x[i] - x[i-1]) * (x[i+1] - x[i]) * (x[i+1] - x[i-1])) for i in range(1, len(x) - 1)])
def fd_at_md(key, e, nu_all):
    sp = CubicSpline(MODE[key]["nu"], MODE[key][e]["dfdlB"]); x = np.asarray(nu_all, float); return x[1:-1] / 2 * d2fd(x, sp(x))
dev = {"law": [], "input": [], "film": []}; dev4 = {"law": [], "input": [], "film": []}; sig = []; errs = []
for e in EPS:
    je = EN[e]; nu = np.array(je["nu"]); md = np.array(je["md"]); er = np.array(je["md_err"])
    axB.errorbar(nu, md, yerr=er, fmt="o", ms=2.4, color=COL[e], capsize=1.3, lw=0.6, zorder=5)
    key = LOO.get(e, FINALK); mo = je["models"][key]
    ng = np.array(mo["nu_grid"]); inr = (ng >= je["nu_all"][0] - 1e-9) & (ng <= je["nu_all"][-1] + 1e-9)
    axB.plot(ng[inr], np.array(mo["curve"])[inr], "-" if e in LOO else "-.", color=COL[e], lw=0.9, zorder=4)
    fl = fd_at_md(key, e, je["nu_all"]); axB.plot(nu, fl, "o", ms=2.6, mfc="white", mec=COL[e], mew=0.6, zorder=6)
    dev["law"] += list(fl - md); sig += list(md); errs += list(er)
    fi, ff = fd_at_md(INPUTK, e, je["nu_all"]), fd_at_md(FILMK, e, je["nu_all"]); dev["input"] += list(fi - md); dev["film"] += list(ff - md)
    if e == "4":
        axB.plot(ng[inr], np.array(je["models"][INPUTK]["curve"])[inr], ":", color=COL[e], lw=0.9, zorder=3)
        axB.plot(ng[inr], np.array(je["models"][FILMK]["curve"])[inr], "--", color=COL[e], lw=0.7, zorder=3)
        axB.plot(nu, fi, "^", ms=2.6, mfc="white", mec=COL[e], mew=0.6, zorder=6); axB.plot(nu, ff, "s", ms=2.3, mfc="white", mec=COL[e], mew=0.6, zorder=6)
        dev4["law"] = list(fl - md); dev4["input"] = list(fi - md); dev4["film"] = list(ff - md)
    M["stats"][f"energy {e}"] = dict(nu=nu.tolist(), MD=md.tolist(), err=er.tolist(), law_fd=fl.tolist(), model_key=key, input_fd=fi.tolist(), film_fd=ff.tolist(),
                                     md_zero_nu=je.get("md_zero_nu"), model_zero_nu=mo.get("zero_nu"))
rmsf = lambda v: float(np.sqrt(np.mean(np.square(v))))
M["energy_rms_fd"] = {k: rmsf(v) for k, v in dev.items()}; M["energy_rms_fd_eps4"] = {k: rmsf(v) for k, v in dev4.items()}
M["energy_rms_md_error"] = rmsf(errs); M["energy_median_md_error"] = float(np.median(errs)); M["energy_signal_rms"] = rmsf(sig); M["energy_n"] = len(sig)
axB.axhline(0, color=INK2, lw=0.5, ls=":"); axB.set_xlabel(r"$\bar n\sigma^3$", labelpad=1); axB.set_ylabel(r"$\partial\Gamma(0)/\partial l_B\ \ (\sigma^{-1})$", labelpad=1)
axB.set_xlim(0.005 if SS.CSET == "01" else 0.015, 0.052); axB.set_ylim(-0.13, 0.36)
axB.legend(handles=[Line2D([], [], color=INK2, marker="o", ms=2.4, lw=0, label="MD"), Line2D([], [], color=INK2, marker="o", ms=2.6, mfc="white", mew=0.6, lw=0, label="model, same difference")] + LSTY
                    + [Line2D([], [], color=COL["4"], lw=0.9, ls=":", marker="^", ms=2.6, mfc="white", mew=0.6, label="network input ($\\varepsilon_r=4$)"),
                       Line2D([], [], color=COL["4"], lw=0.7, ls="--", marker="s", ms=2.3, mfc="white", mew=0.6, label="modulation ($\\varepsilon_r=4$)")],
           loc="upper left", handlelength=2.2, handletextpad=0.5, labelspacing=0.15, borderaxespad=0.2, fontsize=4.8); panel(axB, "b", x=-0.24, y=1.04)
lb = np.array([LB[e] for e in EPS])
for kk, mk, lab in (("k=2", "s", r"$k=2\,\sigma^{-1}$"), ("k=1", "o", r"$k=1\,\sigma^{-1}$")):
    md = -np.array([CH[f"{e}/0.08"][kk]["md_ZZ"] for e in EPS]); mo = -np.array([CH[f"{e}/0.08"][kk]["ZZ"] for e in EPS])
    axC.plot(lb, md, mk, ms=2.6, color=C_MD, lw=0, zorder=5); axC.plot(lb, mo, "-", color=C_FUN, lw=0.9, zorder=4)
    M["stats"][f"-ZZcorr c0.08 {kk}"] = dict(lB=lb.tolist(), MD=md.tolist(), model=mo.tolist(), ratio_over_lB_spread=float(md.max() / lb[np.argmax(md)] / (md / lb).min()))
    a = float(md[0] / lb[0]); lg = np.array([6.5, 33]); axC.plot(lg, a * lg, "--", color=INK2, lw=0.6, zorder=2)
    M["stats"][f"lB-line slope {kk}"] = a
    axC.text(lb[0] * 0.93, md[0], lab, fontsize=5.5, color=INK2, ha="right", va="center")
axC.set_xscale("log"); axC.set_yscale("log"); axC.set_xlim(5.0, 34)
axC.set_xticks([8, 15, 30]); axC.set_xticklabels(["8", "15", "30"]); axC.set_ylim(2.5, 40); axC.set_yticks([5, 10, 20, 40]); axC.set_yticklabels(["5", "10", "20", "40"]); axC.minorticks_off()
axC.set_xlabel(r"$l_B/\sigma$", labelpad=1); axC.set_ylabel(r"$-\Gamma^{\rm corr}_{ZZ}$", labelpad=1)
axC.legend(handles=[Line2D([], [], color=C_MD, marker="s", ms=2.6, lw=0, label="MD"), Line2D([], [], color=C_FUN, lw=0.9, label="functional"),
                    Line2D([], [], color=INK2, lw=0.6, ls="--", label=r"$\propto l_B$")],
           loc="upper left", handlelength=1.6, handletextpad=0.5, labelspacing=0.2, borderaxespad=0.2); panel(axC, "c", x=-0.24, y=1.04)
for c, mk in ((0.08, "s"), (SS.CONCS[0], "o")):
    md = np.array([GMD[e][c][0] for e in EPS]); er = np.array([GMD[e][c][1] for e in EPS]); mo = np.array([gmod(FINALK, e, c) for e in EPS])
    axD.errorbar(lb, md, yerr=er, fmt=mk, ms=2.6, color=C_MD, capsize=1.3, lw=0.6, zorder=5); axD.plot(lb, mo, "-", color=C_FUN, lw=0.9, zorder=4)
    axD.text(lb[-1] + 0.8, md[-1] * (1.12 if c < 0.05 else 1.0), f"$c={c:g}$", fontsize=5.5, color=INK2, va="center")
    M["stats"][f"Gamma(k1) vs lB c{c:g}"] = dict(lB=lb.tolist(), MD=md.tolist(), err=er.tolist(), model=mo.tolist())
axD.axhline(1, color=INK2, lw=0.5, ls=":"); axD.set_xlabel(r"$l_B/\sigma$", labelpad=1); axD.set_ylabel(r"$\Gamma(k_1)$", labelpad=1); axD.set_xlim(6, 36); axD.set_yscale("log")
axD.set_yticks([0.5, 1, 2, 5, 10]); axD.set_yticklabels(["0.5", "1", "2", "5", "10"]); axD.minorticks_off(); panel(axD, "d", x=-0.24, y=1.04)
E4 = ["5", "4", "3", "2"]; CF = 0.08
PRED = {e: np.load(f"{X}/runs/learn/{LOO[e]}/heldout.npz") for e in ("5", "4", "3")}
PE2 = np.load(f"{X}/runs/learn/{FINALK}/predict_c008_E2/predicted.npz")
def pred_of(e, tag):
    if e in PRED: return PRED[e][f"transfer_e{e}_c{CF:g}_{tag}"]
    k = [f for f in PE2.files if f.endswith(f"c{CF:g}_{tag}")]; assert len(k) == 1, (k, PE2.files[:5]); return PE2[k[0]]
for ax, tag, let, chan in ((axE, "p13", "e", "N"), (axF, "p07", "f", "Z")):
    for e in E4:
        P = np.load(f"{EPS_ROOTS[e]}/runs/prod_T1.0_c{CF:g}/field_{tag}/profiles.npz"); x = P["cation_z"] / float(P["lz"]); nb = float(np.mean(P["cation_n"]))
        pr = pred_of(e, tag); sg = 1 if chan == "N" else -1
        ymd = (P["cation_n"] + sg * P["anion_n"]) / (2 * nb); ymo = (pr[0] + sg * pr[1]) / (2 * nb)
        ax.plot(x, ymd, "-", color=COL[e], lw=2.0, alpha=0.35, solid_capstyle="butt", zorder=2)
        ax.plot(x, ymo, "-" if e in LOO else "-.", color=COL[e], lw=0.8, zorder=4)
        M["stats"][f"profile c0.08 {tag} {chan} eps{e}"] = dict(rms_response_MD=float(np.sqrt(np.mean(np.square(ymd - (1 if chan == "N" else 0))))),
                                                                rel_err=float(np.linalg.norm(ymo - ymd) / np.linalg.norm(ymd - (1 if chan == "N" else 0))), model="LOO" if e in LOO else "all-five")
    ax.set_xlim(0, 1); ax.set_xlabel(r"$z/L$", labelpad=1)
    ax.set_ylabel({"N": r"$(n_++n_-)/2\bar n$", "Z": r"$(n_+-n_-)/2\bar n$"}[chan], labelpad=1); panel(ax, let, x=-0.24, y=1.04)
    ax.set_title({"N": r"neutral potential, $c=0.08$", "Z": r"charged potential, $c=0.08$"}[chan], loc="left", pad=2, fontsize=6)
axF.legend(handles=[Line2D([], [], color=COL[e], lw=1.2, label=f"$\\varepsilon_r={e}$") for e in E4] + [Line2D([], [], color=INK2, lw=2.0, alpha=0.35, label="MD")],
           loc="upper center", ncol=3, handlelength=1.2, handletextpad=0.4, labelspacing=0.15, columnspacing=0.8, borderaxespad=0.2, fontsize=5)
axF.set_ylim(-0.55, 0.9); axE.set_ylim(0.3, 1.5)
save(fig, "fig6"); json.dump(M, open(os.path.join(ROOT, "results/figures/fig6_metrics.json"), "w"), indent=1, default=float)
print({k: v for k, v in M.items() if k != "stats"})
