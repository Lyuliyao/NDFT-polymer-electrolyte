"""Protocol driver (spec §5) and evaluation (spec §6).

    python -m learn.protocol v1              [--conc 0.04]
    python -m learn.protocol baselines       (PB and Avni, every concentration)
    python -m learn.protocol train --variant v3 --depth 1 --C 4 --conc 0.04
    python -m learn.protocol learning-curve  [--sizes 4 8 12 all]
    python -m learn.protocol depth-scan      [--depths 1 2 3 --channels 4 8]
    python -m learn.protocol joint           --depth auto --C auto
    python -m learn.protocol transfer        --depth auto --C auto
    python -m learn.protocol eval --model runs/learn/<name>
    python -m learn.protocol summary
    python -m learn.protocol full            (everything above, in order)

Every experiment writes runs/learn/<name>/{config,metrics}.json, params.pkl,
history.npy, heldout.npz and plots/.
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import jax
import jax.numpy as jnp

from . import data as D
from .models import ModelConfig, init_params, force_density
from .train import train, fit_v1, chi2_per_bin
from .evaluate import heldout_report, summarize, bulk_structure, stillinger_lovett
from . import plots

OUT = os.path.join(D.ROOT, "runs", "learn")
_LOG = None
NAME_SUFFIX = ""          # --name-suffix: appended to every experiment name (a new data set, say)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if _LOG is not None:
        _LOG.write(s + "\n")
        _LOG.flush()


def to_np(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def to_jnp(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), tree)


def save_model(d, cfg, params, metrics, hist=None, extra=None):
    os.makedirs(d, exist_ok=True)
    json.dump(asdict(cfg), open(os.path.join(d, "config.json"), "w"), indent=2)
    pickle.dump(to_np(params), open(os.path.join(d, "params.pkl"), "wb"))
    json.dump(metrics, open(os.path.join(d, "metrics.json"), "w"), indent=2, default=float)
    if hist is not None:
        np.save(os.path.join(d, "history.npy"), hist)
    if extra:
        np.savez_compressed(os.path.join(d, "heldout.npz"), **extra)


def load_model(d):
    c = json.load(open(os.path.join(d, "config.json")))
    for k in ("readout_hidden", "grad_channels", "pack_v0", "cace_scale", "loc_hidden", "kernel_net", "cnn_channels"):
        if k in c:
            c[k] = tuple(c[k])
    cfg = ModelConfig(**c)
    params = to_jnp(pickle.load(open(os.path.join(d, "params.pkl"), "rb")))
    return cfg, params


# --------------------------------------------------------------- evaluation
def structure_report(params, cfg, sps, concs=None, k_max=None):
    """S(k), Gamma, Stillinger-Lovett and stability at every state point with
    a zero-field Sk.npz (the model is bulk, so this includes concentrations it
    never saw)."""
    out = {}
    for c, sp in sorted(sps.items()):
        if concs is not None and c not in concs:
            continue
        sk = sp.sk
        if sk is None:
            continue
        N = int(round(sp.L / 0.1))
        g = D.geometry_of(sp, N)
        from .models import cnn_applicable
        if not cnn_applicable(cfg, g):
            continue                       # the CNN read-out does not fit this box's grid
        st = bulk_structure(params, cfg, g, k_md=sk["k"])
        A, _ = stillinger_lovett(params, cfg, g)
        st["SL_A"] = A
        n2 = cfg.n_species * float(sk["n_each"])
        for comp in [x for x in ("ZZ", "NN", "pp", "mm", "pm") if f"S_{x}" in sk and x in st["S_md_k"]]:
            md, mo = sk[f"S_{comp}"] / n2, st["S_md_k"][comp] / n2
            st[f"rel_rms_{comp}"] = float(np.sqrt(np.mean((mo - md) ** 2)) / np.sqrt(np.mean(md ** 2)))
        out[c] = st
    return out


def structure_summary(struct):
    return {f"{c:g}": dict(Gamma=st["Gamma"], SL_A=st["SL_A"], lambda_min_rel=st["lambda_min_rel"],
                           k_lambda_min=st["k_lambda_min"], n_unstable=st["n_unstable"], k_unstable=st["k_unstable"],
                           **{k: st[k] for k in st if k.startswith("rel_rms_")})
            for c, st in struct.items()}


def evaluate(params, cfg, sps, train_runs, heldout_runs, extra_runs=None, name="", outdir=None,
             el=True, ref_params=None):
    """Everything of spec §6 for one model."""
    t0 = time.time()
    metrics = {"name": name, "variant": cfg.variant, "depth": cfg.depth, "C": cfg.C,
               "receptive_field": cfg.receptive_field,
               "train_runs": [r.key for r in train_runs], "heldout_runs": [r.key for r in heldout_runs]}
    btr = D.make_batches(train_runs, sps) if train_runs else {}
    if btr:
        metrics["train_chi2"] = float(chi2_per_bin(params, cfg, [b for b, _ in btr.values()]))
        rows_tr = heldout_report(params, cfg, btr, el=False, log=None)
        metrics["train_rows"] = [{k: v for k, v in r.items() if k != "n_pred"} for r in rows_tr]
    rows_ho, rows_ex = [], []
    extra = {}
    if heldout_runs:
        log(f"  held-out runs ({name}):")
        bho = D.make_batches(heldout_runs, sps)
        rows_ho = heldout_report(params, cfg, bho, el=el, log=log)
        s = summarize(rows_ho)
        metrics["heldout"] = s
        metrics["heldout_chi2"] = s["chi2"]
        if el:
            metrics["heldout_rel_l2"] = float(np.mean([s[f"{lab}_rel_l2"] for lab in ("cation", "anion") if f"{lab}_rel_l2" in s]))
        metrics["heldout_rows"] = [{k: v for k, v in r.items() if k != "n_pred"} for r in rows_ho]
        for r in rows_ho:
            if "n_pred" in r:
                extra[f"c{r['conc']:g}_{r['tag']}"] = r["n_pred"]
    if extra_runs:
        log(f"  transfer runs ({name}):")
        bex = D.make_batches(extra_runs, sps)
        rows_ex = heldout_report(params, cfg, bex, el=el, log=log)
        s = summarize(rows_ex)
        metrics["transfer"] = s
        metrics["transfer_rows"] = [{k: v for k, v in r.items() if k != "n_pred"} for r in rows_ex]
        for r in rows_ex:
            if "n_pred" in r:
                extra[f"transfer_c{r['conc']:g}_{r['tag']}"] = r["n_pred"]
    struct = structure_report(params, cfg, sps)
    metrics["structure"] = structure_summary(struct)
    log("  bulk structure: " + "  ".join(
        f"c={c:g}: Gamma {st['Gamma']:.2f} SL {st['SL_A']-1:+.1e} dS_ZZ {st.get('rel_rms_ZZ', float('nan')):.3f} "
        f"dS_NN {st['rel_rms_NN']:.3f} lmin {st['lambda_min_rel']:.2f}@k={st['k_lambda_min']:.2f}" for c, st in struct.items()))
    metrics["eval_seconds"] = time.time() - t0
    if outdir:
        pd = os.path.join(outdir, "plots")
        os.makedirs(pd, exist_ok=True)
        runs_by_key = {r.key: r for r in train_runs + heldout_runs + (extra_runs or [])}
        ref_rows = None
        if ref_params is not None and heldout_runs:
            ref_rows = heldout_report(ref_params[1], ref_params[0], D.make_batches(heldout_runs, sps), el=el, log=None)
        if rows_ho:
            plots.plot_profiles(rows_ho, runs_by_key, os.path.join(pd, "heldout_profiles.png"), ref_rows, title=name)
            fp = {}
            for c, (b, rs) in D.make_batches(heldout_runs, sps).items():
                pred = np.asarray(jax.vmap(lambda n: force_density(params, cfg, n, b.geom))(b.n))
                for i, r in enumerate(rs):
                    fp[r.key] = pred[i]
            plots.plot_forces(rows_ho, runs_by_key, fp, os.path.join(pd, "heldout_forces.png"), title=name)
        if rows_ex:
            plots.plot_profiles(rows_ex, runs_by_key, os.path.join(pd, "transfer_profiles.png"), title=name + " transfer")
        refs = {}
        if ref_params is not None:
            refs["PB"] = structure_report(ref_params[1], ref_params[0], sps)
        plots.plot_structure(struct, sps, os.path.join(pd, "structure.png"), refs=refs, title=name)
        plots.plot_Wk(struct, os.path.join(pd, "Wk.png"), title=name)
    return metrics, extra


# --------------------------------------------------------------- experiments
def run_experiment(name, cfg, sps, train_runs, heldout_runs, extra_runs=None, opts=None, el=True,
                   val_runs=None):
    """val_runs, when given, drive early stopping instead of the held-out runs,
    which are then used for scoring only."""
    opts = opts or {}
    if val_runs:
        name = name + f"_val{len(val_runs) // max(len({r.conc for r in val_runs}), 1)}"
    name = name + NAME_SUFFIX
    d = os.path.join(OUT, name)
    log(f"=== {name}: {cfg.variant} L={cfg.depth} C={cfg.C} rf~{cfg.receptive_field:.1f}  "
        f"train {len(train_runs)} runs {sorted({r.conc for r in train_runs})}, held-out {len(heldout_runs)}"
        + (f", validation {len(val_runs)}" if val_runs else ""))
    hist = None
    t0 = time.time()
    if cfg.variant == "v1" and not cfg.kernel_net:
        # 2026-10-08: a pair closure with learned kernels (kernel_net) is nonlinear in its parameters and is trained below like the other forms
        btr = D.make_batches(train_runs, sps, kboost=opts.get("kboost", 0.0), kboost_mode=opts.get("kboost_mode", "long12"))
        bl = [b for b, _ in btr.values()]
        ridge = opts.get("ridge", 0.0)
        if ridge == "auto":
            # the smallest ridge (relative to the column scale) for which
            # nbar^-1 + W(k) is positive at every k of every training state point
            for ridge in [0.0] + list(np.geomspace(1e-4, 10.0, 26)):
                params, chi2, cond = fit_v1(cfg, bl, ridge=ridge)
                if all(bulk_structure(params, cfg, b.geom)["n_unstable"] == 0 for b in bl):
                    break
            log(f"  ridge chosen automatically: {ridge:.2e}")
        else:
            params, chi2, cond = fit_v1(cfg, bl, ridge=float(ridge))
        log(f"  least squares (ridge {ridge}): train chi2/bin {chi2:.4f}, condition number {cond:.2e}")
        c = cfg.pair_scale * np.asarray(params["pair"])
        log("  " + ("\n  ".join(f"c({lab}) {c[i].round(2)}" for i, lab in enumerate(("++", "--", "+-"))) if len(c) == 3 else f"c {c[0].round(2)}"))
    elif cfg.variant in ("pb", "avni"):
        params = init_params(cfg)
    else:
        if cfg.variant == "cnn":
            from .models import cnn_pooled_len
            lens = {cnn_pooled_len(cfg, b.geom.z.shape[0]) for b, _ in D.make_batches(train_runs, sps).values()}
            assert len(lens) == 1, f"cnn: the training grids pool to different lengths {lens}"
            cfg = replace(cfg, cnn_len=lens.pop())
            log(f"  CNN read-out on {cfg.cnn_len} x {cfg.cnn_channels[-1]} pooled features")
        if cfg.variant == "cace" and not cfg.cace_scale:
            from .models import cace_feature_scale
            cfg = replace(cfg, cace_scale=cace_feature_scale(
                cfg, [b for b, _ in D.make_batches(train_runs, sps).values()]))
            log(f"  CACE invariants: {len(cfg.cace_scale)}, scales from the training runs "
                f"{np.round(cfg.cace_scale, 1).tolist()}")
        params = init_params(cfg)
        log(f"  parameters: {sum(int(np.size(x)) for x in jax.tree_util.tree_leaves(params))}")
        btr = [b for b, _ in D.make_batches(train_runs, sps, kboost=opts.get("kboost", 0.0),
                                            kboost_mode=opts.get("kboost_mode", "long12")).values()]
        mon = val_runs if val_runs else heldout_runs
        bho = [b for b, _ in D.make_batches(mon, sps).values()] if mon else []
        geoms = [b.geom for b in btr]
        fns = {}
        if opts.get("loss", "force") == "lmu":
            from .signals import lmu_loss
            fns["data_fn"] = lambda p: lmu_loss(p, cfg, btr)
            if bho:
                fns["monitor_fn"] = lambda p: (lambda l: l[0] / l[1])(lmu_loss(p, cfg, bho))
        elif opts.get("loss", "force") == "pcm":
            from .signals import PCMTarget, pcm_loss
            tg = [PCMTarget(sps[round(b.geom.conc, 4)], b.geom) for b in btr]
            log(f"  pair-correlation matching on {len(tg)} state points, {sum(int(t.S.size) for t in tg)} S_ab(k) values; no field run used")
            fns["data_fn"] = lambda p: pcm_loss(p, cfg, tg)
            fns["monitor_fn"] = lambda p: (lambda l: l[0] / l[1])(pcm_loss(p, cfg, tg))
        params, hist, best = train(cfg, params, btr, bho, geoms, log=log, **fns,
                                   **{k: opts[k] for k in ("steps", "lr", "weight_decay", "stab_weight",
                                                            "eval_every", "patience", "select") if k in opts})
    fit_s = time.time() - t0
    pb = ModelConfig("pb", n_species=cfg.n_species); ref = (pb, init_params(pb))
    metrics, extra = evaluate(params, cfg, sps, train_runs, heldout_runs, extra_runs, name=name,
                              outdir=d, el=el, ref_params=ref)
    metrics["fit_seconds"] = fit_s
    metrics["opts"] = opts
    if val_runs:
        metrics["val_runs"] = [r.key for r in val_runs]
        metrics["val_chi2"] = float(chi2_per_bin(params, cfg, [b for b, _ in D.make_batches(val_runs, sps).values()]))
    if hist is not None and len(hist):
        metrics["best_monitor"] = float(best)
        metrics["steps_run"] = int(hist[-1, 0]) + 1
        plots.plot_history(hist, os.path.join(d, "plots", "history.png"), title=name)
    save_model(d, cfg, params, metrics, hist, extra)
    log(f"  -> {d}  train chi2 {metrics.get('train_chi2', float('nan')):.3f}  "
        f"held-out chi2 {metrics.get('heldout_chi2', float('nan')):.3f}  "
        f"rel L2 {metrics.get('heldout_rel_l2', float('nan')):.3f}  ({fit_s:.0f}s fit, {metrics['eval_seconds']:.0f}s eval)")
    return metrics, params


def load_split(concs, args):
    """(state points, all runs, training, held-out, validation) for `concs`;
    validation is empty unless --nval is given."""
    sps, runs = D.load_all()
    ho = D.heldout_tags(runs, regenerate=args.regenerate_splits)
    tr, hold = D.split_runs(runs, ho)
    sel = lambda rs: [r for r in rs if any(abs(r.conc - c) < 1e-6 for c in concs)]
    tr, hold = sel(tr), sel(hold)
    val = []
    if getattr(args, "nval", 0) > 0:
        val, tr = D.validation_split(tr, args.nval)
    return sps, runs, tr, hold, val


def train_opts(args):
    return dict(steps=args.steps, lr=args.lr, weight_decay=args.weight_decay,
                stab_weight=args.stab_weight, eval_every=args.eval_every, patience=args.patience,
                kboost=args.kboost, kboost_mode=args.kboost_mode, loss=args.loss, select=args.select)


def subset(runs, k, seed=D.SPLIT_SEED):
    """Nested random subsets of the training runs (same seed for every size)."""
    order = np.random.default_rng(seed).permutation(len(runs))
    return [runs[i] for i in sorted(order[:k])]


def chosen_depth(args):
    """(depth, C) from the flags, or the best held-out chi2 of the depth scan."""
    if args.depth != "auto":
        return int(args.depth), int(args.C)
    f = os.path.join(OUT, f"depth_scan_c{args.conc[0]:g}".replace(".", ""), "summary.json")
    s = json.load(open(f))
    best = min(s["points"], key=lambda p: p["heldout_chi2"])
    log(f"  chosen depth from {f}: L={best['depth']} C={best['C']} (held-out chi2 {best['heldout_chi2']:.3f})")
    return best["depth"], best["C"]


def ctag(concs):
    return "_".join(f"c{c:g}".replace(".", "") for c in concs)


# ------------------------------------------------------------------ commands
def cmd_v1(args):
    sps, runs, tr, ho, val = load_split(args.conc, args)
    kb = {"kboost": args.kboost, "kboost_mode": args.kboost_mode}
    cfg = ModelConfig("v1", M=args.M, n_species=D.n_species(runs))
    run_experiment(f"v1_{ctag(args.conc)}", cfg, sps, tr, ho, opts={"ridge": 0.0, **kb})
    run_experiment(f"v1ridge_{ctag(args.conc)}", cfg, sps, tr, ho, opts={"ridge": "auto", **kb})


def cmd_baselines(args):
    sps, runs, tr, ho, val = load_split(args.conc, args)
    for v in ("pb", "avni"):
        run_experiment(f"{v}_{ctag(args.conc)}", ModelConfig(v, avni_a=args.avni_a, n_species=D.n_species(runs)), sps, tr, ho, opts={})


def cmd_train(args):
    sps, runs, tr, ho, val = load_split(args.conc, args)
    if args.ntrain:
        tr = subset(tr, args.ntrain)
    cfg = ModelConfig(args.variant, M=args.M, depth=int(args.depth), C=int(args.C),
                      grad_invariants=args.grad_invariants, deep_s_max=args.deep_s_max, seed=args.seed,
                      log_inputs=args.log_inputs, pack_v0=tuple(args.pack_v0), pack_vmax=args.pack_vmax,
                      n_species=D.n_species(runs))
    name = args.name or f"{args.variant}_L{cfg.depth}_C{cfg.C}_{ctag(args.conc)}" + (f"_n{args.ntrain}" if args.ntrain else "")
    run_experiment(name, cfg, sps, tr, ho, opts=train_opts(args), val_runs=val)


def cmd_learning_curve(args):
    sps, runs, tr, ho, val = load_split(args.conc, args)
    sizes = [len(tr) if s == "all" else int(s) for s in args.sizes]
    points = []
    for k in sizes:
        k = min(k, len(tr))
        cfg = ModelConfig("v3", M=args.M, depth=1, seed=args.seed, n_species=D.n_species(runs))
        m, _ = run_experiment(f"lc_{ctag(args.conc)}_n{k}", cfg, sps, subset(tr, k), ho, opts=train_opts(args))
        points.append((k, f"n={k}", m))
    d = os.path.join(OUT, f"learning_curve_{ctag(args.conc)}")
    os.makedirs(d, exist_ok=True)
    json.dump({"points": [dict(n=p[0], heldout_chi2=p[2]["heldout_chi2"], train_chi2=p[2]["train_chi2"],
                               heldout_rel_l2=p[2].get("heldout_rel_l2")) for p in points]},
              open(os.path.join(d, "summary.json"), "w"), indent=2)
    plots.plot_scan(points, os.path.join(d, "learning_curve.png"), "training runs", "V3, L=1", logx=True)
    log("learning curve: " + "  ".join(f"n={p[0]}: ho chi2 {p[2]['heldout_chi2']:.3f}" for p in points))


def cmd_depth_scan(args):
    sps, runs, tr, ho, val = load_split(args.conc, args)
    points = []
    for L in args.depths:
        for C in (args.channels if L > 1 else [args.channels[0]]):
            cfg = ModelConfig("v3", M=args.M, depth=L, C=C, deep_s_max=args.deep_s_max, seed=args.seed, n_species=D.n_species(runs))
            m, _ = run_experiment(f"scan_{ctag(args.conc)}_L{L}_C{C}", cfg, sps, tr, ho, opts=train_opts(args))
            points.append((len(points) + 1, f"L{L}C{C}", dict(m, depth=L, C=C)))
    d = os.path.join(OUT, f"depth_scan_{ctag(args.conc)}")
    os.makedirs(d, exist_ok=True)
    json.dump({"points": [dict(depth=p[2]["depth"], C=p[2]["C"], heldout_chi2=p[2]["heldout_chi2"],
                               train_chi2=p[2]["train_chi2"], heldout_rel_l2=p[2].get("heldout_rel_l2"),
                               receptive_field=p[2]["receptive_field"]) for p in points]},
              open(os.path.join(d, "summary.json"), "w"), indent=2)
    plots.plot_scan(points, os.path.join(d, "depth_scan.png"), "configuration", "V3 depth scan")
    log("depth scan: " + "  ".join(f"{p[1]}: ho chi2 {p[2]['heldout_chi2']:.3f}" for p in points))


def frac_subset(runs, frac, seed=D.SPLIT_SEED):
    """Nested, stratified subsets of the training runs: per concentration the
    runs are ordered round-robin over the strata (random order inside each),
    and the first ceil(frac n) are kept, so a smaller fraction is always
    contained in a larger one."""
    out = []
    for c in sorted({r.conc for r in runs}):
        rs = [r for r in runs if r.conc == c]
        rng = np.random.default_rng([seed, 11, int(round(c * 10000))])
        by = {}
        for r in rs:
            by.setdefault(D._strata(r), []).append(r)
        queues = [[g[i] for i in rng.permutation(len(g))] for _, g in sorted(by.items())]
        order = []
        while any(queues):
            for q in queues:
                if q:
                    order.append(q.pop(0))
        out += order[:max(1, int(np.ceil(frac * len(rs))))]
    return out


def augment_runs(runs, k, reflect, seed=0):
    """k circular shifts (random, seeded) of every run, and their mirror images if reflect.  A shift or a
    reflection of the whole periodic system is an equally valid equilibrium sample; the z component of
    the force density changes sign under reflection."""
    rng = np.random.default_rng([seed, 97]); out = list(runs)
    for r in runs:
        N = r.n.shape[1]
        shifts = [0] + [int(x) for x in rng.integers(1, N, size=k)]
        for j, sft in enumerate(shifts):
            roll = lambda a: np.roll(a, sft, axis=-1)
            if j > 0:
                out.append(replace(r, tag=f"{r.tag}_sh{sft}", n=roll(r.n), sig_n=roll(r.sig_n), f=roll(r.f),
                                   sig_f=roll(r.sig_f), V=roll(r.V), w=roll(r.w)))
            if reflect:
                fl = lambda a: np.roll(a, sft, axis=-1)[:, ::-1].copy()
                out.append(replace(r, tag=f"{r.tag}_sh{sft}_R", n=fl(r.n), sig_n=fl(r.sig_n), f=-fl(r.f),
                                   sig_f=fl(r.sig_f), V=fl(r.V), w=fl(r.w)))
    return out


def cmd_joint(args):
    L, C = chosen_depth(args)
    sps, runs, tr, ho, val = load_split(args.joint_conc, args)
    if args.train_frac < 1.0:
        n0 = len(tr)
        tr = frac_subset(tr, args.train_frac)
        log(f"  training subset: fraction {args.train_frac:g}, {len(tr)} of {n0} runs "
            f"({', '.join(f'{c:g}: {sum(r.conc == c for r in tr)}' for c in sorted({r.conc for r in tr}))})")
    if args.augment_shifts > 0 or args.augment_reflect:
        n0 = len(tr)
        tr = augment_runs(tr, args.augment_shifts, args.augment_reflect, seed=args.seed)
        log(f"  augmented training set: {n0} runs -> {len(tr)} ({args.augment_shifts} shifts each"
            f"{', with mirror images' if args.augment_reflect else ''})")
    for v in args.variants:
        cfg = ModelConfig(v, M=args.M, depth=L, C=C, grad_invariants=args.grad_invariants, seed=args.seed,
                          hidden=args.hidden, readout_hidden=tuple(args.readout_hidden),
                          win_bins=args.win_bins, cace_a=args.cace_a, cace_qcut=args.cace_qcut,
                          loc_hidden=tuple(args.loc_hidden), s_max=args.s_max, kernel_net=tuple(args.kernel_net),
                          activation=args.activation, n_ref=args.n_ref, cnn_dilation=args.cnn_dilation,
                          cnn_channels=tuple(args.cnn_channels), cnn_rows=args.cnn_rows, n_species=D.n_species(runs))
        wide = "" if tuple(args.readout_hidden) == (64, 64) and args.hidden == 64 else f"_R{'x'.join(str(h) for h in args.readout_hidden)}_H{args.hidden}"
        arch = {"c1win": f"_W{args.win_bins}", "cace": f"_a{args.cace_a}q{args.cace_qcut}",
                "cnn": f"_d{args.cnn_dilation}"}.get(v, "")
        run_experiment(f"joint_{v}{arch}_L{L}_C{C}{wide}_{ctag(args.joint_conc)}", cfg, sps, tr, ho, opts=train_opts(args),
                       val_runs=val)


def cmd_transfer(args):
    L, C = chosen_depth(args)
    sps, runs, tr_all, ho_all, val_all = load_split(args.joint_conc, args)
    targets = args.targets or [c for c in args.joint_conc if c != max(args.joint_conc)]
    for target in targets:
        src = [c for c in args.joint_conc if c != target]
        tr = [r for r in tr_all if r.conc in src]
        ho = [r for r in ho_all if r.conc in src]
        val = [r for r in val_all if r.conc in src]
        ex = [r for r in runs if r.status == "PASS" and abs(r.conc - target) < 1e-6]
        for v in args.variants:
            cfg = ModelConfig(v, M=args.M, depth=L, C=C, seed=args.seed, n_species=D.n_species(runs))
            run_experiment(f"transfer_{v}_L{L}_C{C}_{ctag(src)}_to_{ctag([target])}", cfg, sps, tr, ho,
                           extra_runs=ex, opts=train_opts(args), val_runs=val)


def cmd_eval(args):
    cfg, params = load_model(args.model)
    sps, runs = D.load_all()
    m0 = json.load(open(os.path.join(args.model, "metrics.json")))
    keys = lambda ks: [r for r in runs if r.key in ks]
    pb = ModelConfig("pb", n_species=cfg.n_species); ref = (pb, init_params(pb))
    metrics, extra = evaluate(params, cfg, sps, keys(m0["train_runs"]), keys(m0["heldout_runs"]),
                              name=os.path.basename(args.model), outdir=args.model, ref_params=ref)
    metrics.update({k: m0[k] for k in ("fit_seconds", "opts", "best_monitor", "steps_run") if k in m0})
    save_model(args.model, cfg, params, metrics, extra=extra)


def cmd_predict(args):
    """Score a saved model on every PASS run of the given concentrations
    (force residual, Euler-Lagrange profiles, S(k)); nothing is trained.
    Writes runs/learn/<model>/predict_<conc>/."""
    cfg, params = load_model(args.model)
    sps, runs = D.load_all()
    # --status PASS REVIEW scores runs under review as well (size test, 2026-10-05: the aperiodic long-box runs,
    # whose maximum driven-mode pull over 34-43 driven modes exceeds 3 while the median is below 1); logged.
    ex = [r for r in runs if r.status in args.status and any(abs(r.conc - c) < 1e-6 for c in args.conc)]
    if not ex:
        log(f"no {'/'.join(args.status)} runs with profiles at {args.conc}")
        return
    if any(r.status != "PASS" for r in ex):
        log("  scored with status " + ", ".join(f"{r.tag}:{r.status}" for r in ex if r.status != "PASS"))
    name = os.path.basename(args.model.rstrip("/")) + "_predict_" + ctag(args.conc)
    d = os.path.join(args.model, "predict_" + ctag(args.conc))
    pb = ModelConfig("pb", n_species=cfg.n_species); ref = (pb, init_params(pb))
    metrics, extra = evaluate(params, cfg, sps, [], [], extra_runs=ex, name=name, outdir=d, ref_params=ref)
    metrics["model"] = args.model
    os.makedirs(d, exist_ok=True)
    json.dump(metrics, open(os.path.join(d, "metrics.json"), "w"), indent=2, default=float)
    np.savez_compressed(os.path.join(d, "predicted.npz"), **extra)
    t = metrics["transfer"]
    log(f"  -> {d}: {t['n_runs']} runs, chi2 {t['chi2']:.3f}, rel L2 " + "/".join(f"{t[k]:.3f}" for k in ("cation_rel_l2", "anion_rel_l2") if k in t) + ", "
        f"EL converged {t['el_converged']}/{t['n_runs']}")


def cmd_summary(args):
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "*", "metrics.json"))):
        m = json.load(open(f))
        st = m.get("structure", {})
        rows.append((m["name"] + ("" if "val_chi2" not in m else ""), m["variant"], m["depth"], m["C"], len(m["train_runs"]),
                     m.get("train_chi2", np.nan), m.get("heldout_chi2", np.nan), m.get("heldout_rel_l2", np.nan),
                     m.get("transfer", {}).get("chi2", np.nan),
                     " ".join(f"{c}:{v['Gamma']:.2f}" for c, v in st.items()),
                     " ".join(f"{v['rel_rms_ZZ']:.2f}" for v in st.values())))
    hdr = f"{'name':44s} {'var':5s} L C  ntr {'tr chi2':>8s} {'ho chi2':>8s} {'ho L2':>6s} {'tr.chi2':>8s}  Gamma(c)  |  dS_ZZ(c)"
    lines = [hdr] + [f"{r[0]:44s} {r[1]:5s} {r[2]} {r[3]}  {r[4]:3d} {r[5]:8.3f} {r[6]:8.3f} {r[7]:6.3f} {r[8]:8.3f}  {r[9]}  |  {r[10]}" for r in rows]
    log("\n".join(lines))
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(lines) + "\n")


def cmd_full(args):
    cmd_v1(args)
    cmd_baselines(args)
    cmd_learning_curve(args)
    cmd_depth_scan(args)
    cmd_joint(args)
    cmd_transfer(args)
    cmd_summary(args)


def main(argv=None):
    global _LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["v1", "baselines", "train", "learning-curve", "depth-scan", "joint",
                                        "transfer", "eval", "predict", "summary", "full"])
    ap.add_argument("--conc", type=float, nargs="+", default=[0.04])
    ap.add_argument("--status", nargs="+", default=["PASS"], help="predict: run statuses to score (PASS; REVIEW on request)")
    ap.add_argument("--joint-conc", type=float, nargs="+", default=[0.01, 0.02, 0.04])
    ap.add_argument("--targets", type=float, nargs="+", default=None,
                    help="transfer: concentrations to leave out and predict (default: all but the highest)")
    ap.add_argument("--variant", default="v3")
    ap.add_argument("--variants", nargs="+", default=["v2", "v3"])
    ap.add_argument("--depth", default="1")
    ap.add_argument("--C", default="4")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--channels", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=64, help="hidden width of the gating MLPs (depth >= 2)")
    ap.add_argument("--readout-hidden", type=int, nargs="+", default=[64, 64], help="hidden widths of the readout MLP Phi")
    ap.add_argument("--deep-s-max", type=float, default=-1.0)
    ap.add_argument("--grad-invariants", action="store_true")
    ap.add_argument("--avni-a", type=float, default=1.0)
    ap.add_argument("--sizes", nargs="+", default=["4", "8", "12", "all"])
    ap.add_argument("--ntrain", type=int, default=0)
    ap.add_argument("--nval", type=int, default=0, help="validation runs per concentration, taken from the training runs")
    ap.add_argument("--log-inputs", action="store_true")
    ap.add_argument("--pack-v0", type=float, nargs=2, default=[-1.0, -1.0])
    ap.add_argument("--pack-vmax", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--stab-weight", type=float, default=300.0)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--regenerate-splits", action="store_true")
    ap.add_argument("--log", default=None)
    ap.add_argument("--name-suffix", default="")
    ap.add_argument("--loss", choices=["force", "lmu", "pcm"], default="force",
                    help="training signal: force matching (ours), local chemical-potential balance (Cheng 2026), "
                         "or pair-correlation matching on the zero-field S_ab(k) (Dijkman 2025); learn.signals")
    ap.add_argument("--s-max", type=float, default=2.0, help="width of the widest Gaussian of the basis")
    ap.add_argument("--n-ref", type=float, default=0.02, help="density scale of the network inputs")
    ap.add_argument("--activation", choices=["silu", "gelu", "tanh", "softplus"], default="silu",
                    help="activation of the readout Phi and of the kernel network g")
    ap.add_argument("--kernel-net", type=int, nargs="*", default=[],
                    help="hidden widths of g(r): learned radial kernels B_m(r)[1 + g_m(r)]")
    ap.add_argument("--win-bins", type=int, default=30, help="c1win: half-width of the density window in bins")
    ap.add_argument("--cnn-channels", type=int, nargs="+", default=[16, 16, 32, 32, 64, 64],
                    help="cnn: channels of the six dilated periodic convolutions (Dijkman 2025)")
    ap.add_argument("--cnn-dilation", type=int, default=2, help="cnn: dilation of the convolutions")
    ap.add_argument("--cnn-rows", type=int, default=8, help="cnn: Hessian rows averaged for W(k)")
    ap.add_argument("--augment-shifts", type=int, default=0,
                    help="joint: add this many random circular shifts of every training run (symmetry by data)")
    ap.add_argument("--augment-reflect", action="store_true", help="joint: also add the mirror image of every copy")
    ap.add_argument("--cace-a", type=int, default=5, help="cace: voxel size in bins (odd)")
    ap.add_argument("--cace-qcut", type=int, default=3, help="cace: stencil radius in voxels")
    ap.add_argument("--loc-hidden", type=int, nargs="+", default=[32, 16], help="cace: hidden widths of a_loc")
    ap.add_argument("--select", choices=["best", "last"], default="best",
                    help="best: the parameters with the lowest monitor (early stopping); last: the final parameters after all steps")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="joint: keep this fraction of the training runs per concentration (nested, stratified)")
    ap.add_argument("--kboost", type=float, default=0.0,
                    help="extra Fourier-space loss on the force residual, mode m weighted (k_3/k_m)^kboost (see --kboost-mode)")
    ap.add_argument("--kboost-mode", default="long12", choices=["long12", "all", "k"],
                    help="k: k_m^-kboost on every mode of every run (k in 1/sigma); all: the same with (k_3/k_m)^kboost; "
                         "long12: m = 1, 2 of the long-wavelength runs only, total weight (k_3/k_m)^kboost")
    args = ap.parse_args(argv)
    global NAME_SUFFIX
    NAME_SUFFIX = args.name_suffix
    os.makedirs(OUT, exist_ok=True)
    _LOG = open(args.log or os.path.join(OUT, "protocol.log"), "a")
    log(f"\n##### {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(sys.argv[1:])}")
    globals()[f"cmd_{args.command.replace('-', '_')}"](args)


if __name__ == "__main__":
    main()
