"""Fig. 2: predicted density profiles at both coupling strengths.  Rows: eps_r = 7.5 (l_B = 7.96 sigma) and
eps_r = 2 (l_B = 29.8 sigma), each trained on its own data.  Columns: three test runs (c = 0.02 held-out p14,
c = 0.05 never trained p01, c = 0.06 held-out p14; the same potentials at both eps_r), cation density: MD (band
+-2 s.e.), Poisson-Boltzmann, pair closure and the neural functional (seed 0), all solved from V(z) alone.
Profile errors (mean of cation and anion relative L2) of every test run for all models -> fig2_metrics.json.
SI figure si/chi2_test: force residual chi2 per test run at eps_r = 7.5."""
from figstyle import *
from learn import data as D
from learn.models import ModelConfig, init_params
from learn.evaluate import el_solve, profile_metrics
from learn.train import residual_pulls
from learn.protocol import load_model

OUT = "p27"
pb = (ModelConfig("pb"), init_params(ModelConfig("pb")))
SYS, TEST, GEO, PRED, MET = {}, {}, {}, {}, {}
for e in ("7.5", "2"):
    S = system(e); SYS[e] = S
    sps, runs = D.load_all(root=S["root"]); ho = D.heldout_tags(runs, root=S["root"]); test = test_runs(runs, ho)
    geoms = {c: D.geometry_of(sps[c], int(round(sps[c].L / 0.1))) for c in CONCS}
    P = {"PB": {r.key: el_solve(pb[1], pb[0], geoms[r.conc], r.V)[0] for r in test}, "pair": load_preds(S["pair"], test)}
    for s, d in enumerate(S["fun"]):
        P[f"fun{s}"] = load_preds(d, test)
    test = [r for r in test if all(r.key in p for p in P.values())]
    met = {m: {} for m in P}
    for m, pp in P.items():
        for r in test:
            pm = profile_metrics(pp[r.key], r.n)
            met[m][r.key] = 0.5 * (pm["cation"]["rel_l2"] + pm["anion"]["rel_l2"])
    if e == "7.5":     # force residual per test run (SI figure)
        # Replay the weights of the archived fits; plotted bands use joint blocks.
        _, archived = D.load_all(root=S["root"], uncertainty="legacy")
        archived = {r.key: r for r in archived}
        cfg, ps = load_model(S["fun"][0]); chi = {"PB": {}, "fun": {}}
        for c in CONCS:
            rs = [archived[r.key] for r in test if r.conc == c]; b, _ = D.make_batches(rs, sps)[c]
            for lab, (cf, pr) in {"PB": pb, "fun": (cfg, ps)}.items():
                v = (np.asarray(residual_pulls(pr, cf, b)) ** 2).reshape(len(rs), 2, -1).mean(axis=2).mean(axis=1)
                for i, r in enumerate(rs): chi[lab][r.key] = float(v[i])
        CHI, TEST75 = chi, test
    TEST[e], GEO[e], PRED[e], MET[e] = test, geoms, P, met
json.dump({e: MET[e] for e in MET} | {"chi2_7.5": CHI}, open(os.path.join(ROOT, "paper/figures/fig2_metrics.json"), "w"), indent=1)

# ---- figure
H = 92
fig = plt.figure(figsize=(183 * MM, H * MM))
def ax_mm(x, y, w, h): return fig.add_axes([x / 183, y / H, w / 183, h / H])
show = ["c0.02/p14", "c0.05/p01", "c0.06/p14"]
titles = {"c0.02/p14": "$c$ = 0.02, held-out", "c0.05/p01": "$c$ = 0.05, never trained", "c0.06/p14": "$c$ = 0.06, held-out"}
ROWY = {"7.5": 49, "2": 10}; RH = 30; LB = {"7.5": "7.96", "2": "29.8"}
axes = {}
for e in ("7.5", "2"):
    y0 = ROWY[e]; top = e == "7.5"
    for j, key in enumerate(show):
        r = [x for x in TEST[e] if x.key == key][0]; g = GEO[e][r.conc]; z = (np.arange(r.n.shape[1]) + 0.5) * g.dz
        ax = ax_mm(22 + j * 54, y0, 47, RH); axes[(e, j)] = ax
        nb = g.nbar
        ax.fill_between(z, (r.n[0] - 2 * r.sig_n[0]) / nb, (r.n[0] + 2 * r.sig_n[0]) / nb, color=BAND, lw=0)
        ax.plot(z, r.n[0] / nb, color=C_MD, lw=1.0, label="MD")
        ax.plot(z, PRED[e]["PB"][key][0] / nb, color=C_PB, lw=0.9, ls=":", label="Poisson–Boltzmann")
        ax.plot(z, PRED[e]["pair"][key][0] / nb, color=C_PAIR, lw=0.9, ls="--", label="pair closure")
        ax.plot(z, PRED[e]["fun0"][key][0] / nb, color=C_FUN, lw=0.9, label="neural functional")
        ax.set_xlim(0, g.L)
        allc = np.concatenate([r.n[0], PRED[e]["pair"][key][0], PRED[e]["fun0"][key][0]]) / nb
        ax.set_ylim(0, allc.max() * 1.25)
        ax.text(0.98, 0.97, f"pair {100 * MET[e]['pair'][key]:.1f}%  neural {100 * MET[e]['fun0'][key]:.1f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=5.3, color=INK2)
        if top: ax.set_title(titles[key], loc="left", pad=2); ax.set_xticklabels([])
        else: ax.set_xlabel(r"$z/\sigma$", labelpad=1)
        if j == 0: ax.set_ylabel(r"cation $n_+(z)/\bar n$", labelpad=1)
    fig.text(5 / 183, (y0 + RH / 2) / H, f"$\\varepsilon_r$ = {e}\n$l_B$ = {LB[e]} $\\sigma$", rotation=90, ha="center", va="center", fontsize=6.5)
axes[("7.5", 0)].legend(loc="lower left", ncol=4, bbox_to_anchor=(0, 1.2), borderaxespad=0, handlelength=1.8, columnspacing=1.4)
save(fig, "fig2")

# ---- SI: the force residual on the same runs (eps_r = 7.5)
x = np.arange(len(TEST75)); bounds = [i for i in range(1, len(TEST75)) if TEST75[i].conc != TEST75[i - 1].conc]
centers = [np.mean([i for i, r in enumerate(TEST75) if r.conc == c]) for c in CONCS]
fig = plt.figure(figsize=(89 * MM, 52 * MM))
ax = fig.add_axes([16 / 89, 12 / 52, 70 / 89, 30 / 52])
ax.plot(x, [CHI["PB"][r.key] for r in TEST75], "o", ms=2.0, mfc="white", mec=C_PB, mew=0.6, label="Poisson–Boltzmann")
ax.plot(x, [CHI["fun"][r.key] for r in TEST75], "o", ms=2.0, color=C_FUN, label="neural functional")
ax.set_yscale("log"); ax.set_ylabel("force residual $\\chi^2$ per bin", labelpad=1)
for bnd in bounds: ax.axvline(bnd - 0.5, color=BAND, lw=0.5)
ax.axhline(1.0, color="#b0afa9", lw=0.6, ls="--"); ax.set_xlim(-0.8, len(TEST75) - 0.2); ax.set_xticks([]); ax.tick_params(axis="x", length=0)
for i, (c, cx) in enumerate(zip(CONCS, centers)):
    ax.text(cx, -0.03 - 0.06 * (i % 2), f"{c:g}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5, color=INK2)
ax.set_xlabel("test runs, grouped by $c$", labelpad=15)
ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, 1.02), borderaxespad=0, handlelength=1.2)
save(fig, "si/chi2_test")

# ---- numbers for the text
for e in ("7.5", "2"):
    keys = [r.key for r in TEST[e]]
    for m in MET[e]:
        v = 100 * np.array([MET[e][m][k] for k in keys])
        print(f"eps {e} {m:5s} n={len(v)} range {v.min():.2f}-{v.max():.2f}%  mean {v.mean():.2f}%",
              " | by c (mean):", {c: round(float(np.mean([100 * MET[e][m][k] for k in keys if k.startswith(f'c{c:g}/')])), 2) for c in CONCS})
    if e == "2":
        print("  p27:", {m: round(100 * MET[e][m]["c0.04/p27"], 2) for m in MET[e] if "c0.04/p27" in MET[e][m]})
