"""Data interface (spec §1): accepted field runs -> arrays on the bin grid.

A run is used when its verdict is PASS.  The verdict comes from
runs/field_analysis/acceptance_summary.csv (which also carries the YBG check);
a run absent from the CSV falls back to acceptance.json's `pass`.  Runs listed
in splits/excluded.json are EXCLUDED.  Gaussian-train runs are folded over
their period.  The held-out list lives in splits/heldout.json and is written
once, with a fixed seed, never regenerated unless asked.
"""
import csv
import glob
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from .geometry import Geometry, SPECIES, make_geometry, fold_periodic

ROOT = os.environ.get("SIP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT_SEED = 20260925
N_HELDOUT = 4


# ------------------------------------------------------------ state points
@dataclass
class StatePoint:
    conc: float
    path: str
    zero_field: str
    L: float
    n_pairs: int
    lB: float
    volume: float

    @property
    def sk(self):
        f = os.path.join(self.zero_field, "Sk.npz")
        return dict(np.load(f)) if os.path.exists(f) else None


def discover_state_points(root=ROOT):
    """Every runs/* directory with a zero-field D.json.  When two directories
    share a concentration the one with more profiles wins (pilot vs prod)."""
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "runs", "*"))):
        for zf in ("zero_field", "zerofield"):
            dj = os.path.join(d, zf, "D.json")
            if os.path.exists(dj):
                break
        else:
            continue
        D = json.load(open(dj))
        sp = StatePoint(conc=float(D["conc"]), path=d, zero_field=os.path.join(d, zf),
                        L=float(D["lz"]), n_pairs=int(D["n_pairs"]),
                        lB=float(np.load(os.path.join(d, zf, "Sk.npz"))["lB"])
                        if os.path.exists(os.path.join(d, zf, "Sk.npz")) else 7.957187947537492,
                        volume=float(D["volume"]))
        nprof = len(glob.glob(os.path.join(d, "field_p*", "profiles.npz")))
        key = round(sp.conc, 4)
        if key not in out or nprof > out[key][1]:
            out[key] = (sp, nprof)
    return {k: v[0] for k, v in out.items()}


# --------------------------------------------------------------- verdicts
def load_verdicts(root=ROOT):
    f = os.path.join(root, "runs", "field_analysis", "acceptance_summary.csv")
    v = {}
    if os.path.exists(f):
        for r in csv.DictReader(open(f)):
            v[(os.path.basename(r["state"].rstrip("/")), r["tag"])] = r["verdict"]
    return v


def load_excluded(root=ROOT):
    f = os.path.join(root, "splits", "excluded.json")
    if not os.path.exists(f):
        return set()
    ex = json.load(open(f))
    return {(str(k), t) for k, ts in ex.items() for t in ts}


# ------------------------------------------------------------------- runs
@dataclass
class Run:
    conc: float
    tag: str
    family: str
    kind: str
    path: str
    status: str
    fold_q: int
    n: np.ndarray        # (2, N) MD densities
    sig_n: np.ndarray
    f: np.ndarray        # (2, N) internal force density, z component
    sig_f: np.ndarray
    V: np.ndarray        # (2, N) external potential per species
    w: np.ndarray        # (2, N) loss weights 1/(sig_f^2 + eps^2)
    eps: np.ndarray      # (2,) the floor, median sig_f per species
    spec: dict = field(repr=False)
    uncertainty_method: str = "legacy_diagonal"

    @property
    def key(self):
        return f"c{self.conc:g}/{self.tag}"

    @property
    def contrast(self):
        return (self.n.max(1) / np.maximum(self.n.min(1), 1e-12))


def _status(sp, tag, verdicts, excluded, path):
    ck = f"{sp.conc:g}"
    if (ck, tag) in excluded or (os.path.basename(sp.path), tag) in excluded:
        return "EXCLUDED"
    v = verdicts.get((os.path.basename(sp.path), tag))
    if v is not None:
        return "PASS" if v == "PASS" else "REVIEW"
    aj = os.path.join(path, "acceptance.json")
    if os.path.exists(aj):
        return "PASS" if json.load(open(aj)).get("pass") else "REVIEW"
    return "MISSING"


def species_of(d):
    """Species stored in a profiles file: ("cation", "anion") for the ions, ("cation",) for a one-component fluid."""
    return tuple(s for s in SPECIES if f"{s}_n" in d)


def n_species(runs):
    return runs[0].n.shape[0] if runs else 2


def profile_uncertainties(d, q=1, mode="recorded"):
    """Return profile standard errors without assuming independent periods.

    New files store paired time blocks.  ``joint`` requires those blocks;
    ``legacy`` explicitly reproduces the weights used by archived models.
    ``recorded`` uses blocks when present and visibly flags legacy files.
    """
    if mode not in {"recorded", "joint", "legacy"}:
        raise ValueError(f"unknown uncertainty mode: {mode}")
    sps = species_of(d)
    keys = [f"{s}_{k}_blocks" for s in sps for k in ("n", "f_int")]
    if mode != "legacy" and all(k in d for k in keys):
        errors = {}
        for k in ("n", "f_int"):
            b = np.stack([d[f"{s}_{k}_blocks"] for s in sps])
            if b.ndim != 3 or b.shape[1] < 2 or not np.isfinite(b).all():
                raise ValueError("finite profile blocks of shape (n_species, B >= 2, N) required")
            if q != 1:
                b = fold_periodic(b, q)
            errors[k] = b.std(axis=1, ddof=1) / np.sqrt(b.shape[1])
        return errors["n"], errors["f_int"], "paired_blocks_v1"
    if mode == "joint":
        raise ValueError("joint uncertainties require n_blocks and f_int_blocks; regenerate profiles")
    if mode == "recorded":
        warnings.warn("Profile blocks unavailable: retaining legacy diagonal errors; "
                      "regenerate profiles for covariance-aware uncertainties.", RuntimeWarning,
                      stacklevel=2)
    sn = np.stack([d[f"{s}_n_err"] for s in sps])
    sf = np.stack([d[f"{s}_f_int_err_legacy"] if f"{s}_f_int_err_legacy" in d
                   else d[f"{s}_f_int_err"] for s in sps])
    return sn / np.sqrt(q), sf / np.sqrt(q), "legacy_diagonal"


def load_run(sp, tag, verdicts=None, excluded=None, fold=True, uncertainty="recorded"):
    path = os.path.join(sp.path, f"field_{tag}")
    npz = os.path.join(path, "profiles.npz")
    if not os.path.exists(npz):
        return None
    d = np.load(npz, allow_pickle=True)
    spec = json.loads(str(d["pot_json"]))
    lz = float(d["lz"])
    if abs(lz - sp.L) > 1e-6:
        raise ValueError(f"{path}: profile box {lz} != zero-field box {sp.L}")
    z = d["cation_z"]
    N = len(z)
    assert abs(z[1] - z[0] - lz / N) < 1e-9
    sps = species_of(d)
    n = np.stack([d[f"{s}_n"] for s in sps])
    f = np.stack([d[f"{s}_f_int"] for s in sps])
    V = np.stack([d[f"{s}_V"] for s in sps])
    family = spec.get("family", "fourier")
    q = 1
    if fold and family == "gauss":
        q = int(round(lz / spec["gauss"]["period"]))
        n, f, V = fold_periodic(n, q), fold_periodic(f, q), fold_periodic(V, q)
    sn, sf, uncertainty_method = profile_uncertainties(d, q, uncertainty)
    eps = np.median(sf, axis=1)
    w = 1.0 / (sf ** 2 + eps[:, None] ** 2)
    # SIP_WEIGHT: loss-weight test (2026-10-04). "chi2": plain 1/sigma^2; "uniform": one constant
    # for every bin, run and species (1/<eps^2> over this run, so the scale stays comparable).
    wm = os.environ.get("SIP_WEIGHT", "paper")
    if wm == "chi2":
        w = 1.0 / sf ** 2
    elif wm == "uniform":
        w = np.full_like(w, 1.0 / np.mean(eps ** 2))
    elif wm != "paper":
        raise ValueError(f"SIP_WEIGHT={wm}")
    st = _status(sp, tag, verdicts or {}, excluded or set(), path)
    return Run(conc=sp.conc, tag=tag, family=family, kind=spec["kind"], path=path,
               status=st, fold_q=q, n=n, sig_n=sn, f=f, sig_f=sf, V=V, w=w, eps=eps,
               spec=spec, uncertainty_method=uncertainty_method)


def load_all(root=ROOT, concs=None, fold=True, uncertainty="recorded"):
    """All runs with profiles, any status.  Returns (state points, runs)."""
    sps = discover_state_points(root)
    verd, excl = load_verdicts(root), load_excluded(root)
    runs = []
    for c, sp in sorted(sps.items()):
        if concs is not None and not any(abs(c - cc) < 1e-6 for cc in concs):
            continue
        for p in sorted(glob.glob(os.path.join(sp.path, "field_p*"))):
            r = load_run(sp, os.path.basename(p)[6:], verd, excl, fold=fold,
                         uncertainty=uncertainty)
            if r is not None:
                runs.append(r)
    return sps, runs


def geometry_of(sp, N):
    return make_geometry(sp.L, N, sp.n_pairs, sp.lB, sp.conc)


# ----------------------------------------------------------------- splits
def _strata(r):
    if r.family == "gauss":
        return "gauss"
    if r.family == "mixed":
        return "mixed"
    return "fourier_both" if r.kind == "both" else "fourier_one"


def heldout_tags(runs, root=ROOT, regenerate=False, seed=SPLIT_SEED, n_heldout=N_HELDOUT):
    """{conc: [tags]} from splits/heldout.json; written once from PASS runs,
    one per stratum (fourier both / fourier one-part / mixed / gauss)."""
    f = os.path.join(root, "splits", "heldout.json")
    out = {}
    if os.path.exists(f) and not regenerate:
        out = {float(k): v for k, v in json.load(open(f)).items()}
    concs = [c for c in sorted({r.conc for r in runs}) if c not in out
             and any(r.conc == c and r.status == "PASS" for r in runs)]
    if not concs:
        return out
    # existing entries are never touched; a new concentration gets its own
    # draw, seeded by (seed, concentration) so that it does not depend on
    # which other concentrations were present when it was added
    for c in concs:
        rng = np.random.default_rng([seed, int(round(c * 10000))])
        ok = [r for r in runs if r.conc == c and r.status == "PASS"]
        by = {}
        for r in ok:
            by.setdefault(_strata(r), []).append(r.tag)
        chosen = []
        for s in ("fourier_both", "fourier_one", "mixed", "gauss"):
            if by.get(s):
                chosen.append(str(rng.choice(sorted(by[s]))))
        rest = sorted(set(r.tag for r in ok) - set(chosen))
        while len(chosen) < n_heldout and rest:
            chosen.append(str(rng.choice(rest)))
            rest.remove(chosen[-1])
        out[c] = sorted(chosen[:n_heldout])
    os.makedirs(os.path.dirname(f), exist_ok=True)
    json.dump({f"{k:g}": v for k, v in sorted(out.items())}, open(f, "w"), indent=2)
    return out


def validation_split(train_runs, nval, seed=SPLIT_SEED):
    """Take `nval` runs per concentration out of the training runs for early
    stopping, so that the held-out runs are only ever used for scoring.
    Returns (val_runs, remaining_train_runs); deterministic.  The linear-regime
    runs (family "linear", added after the splits were drawn) always stay in
    training, so the draw among the other runs is the original one."""
    val, rest = [], []
    for c in sorted({r.conc for r in train_runs}):
        rest += [r for r in train_runs if r.conc == c and r.family == "linear"]
        rs = [r for r in train_runs if r.conc == c and r.family != "linear"]
        order = np.random.default_rng([seed, 7, int(round(c * 10000))]).permutation(len(rs))
        pick = set(order[:nval].tolist())
        val += [rs[i] for i in sorted(pick)]
        rest += [rs[i] for i in range(len(rs)) if i not in pick]
    return val, rest


def split_runs(runs, heldout):
    tr, ho = [], []
    for r in runs:
        if r.status != "PASS":
            continue
        (ho if r.tag in heldout.get(r.conc, []) else tr).append(r)
    return tr, ho


# ---------------------------------------------------------------- batches
class Batch(NamedTuple):
    """Runs of one state point stacked: arrays (R, 2, N) and the geometry.
    kboost (R, N//2+1) extra weight on each Fourier mode of the force residual (train.batch_loss); the
    residual (1 = plain force matching)."""
    n: jnp.ndarray
    f: jnp.ndarray
    w: jnp.ndarray
    V: jnp.ndarray
    geom: Geometry
    kboost: jnp.ndarray


KBOOST_FAMILIES = ("long", "long3")


def mode_boost(run, N, kboost=0.0, m_ref=3, mode="long12", L=None):
    """Extra loss weight per Fourier mode of the force residual, added to the
    per-bin loss (train.batch_loss).  Force matching weighs a mode by k^2
    (f = -n d_z mu), so the long-wavelength response carries almost no weight
    in the plain loss.  mode "all": (m_ref/m)^kboost on every mode m >= 1 of
    every run, a spectrum loss weighted towards long wavelengths (kboost = 2:
    the modes are weighted as chemical-potential residuals relative to m_ref).
    mode "long12" (the earlier runs): (m_ref/m)^kboost - 1 on m = 1, 2 of the
    long-wavelength runs only, so that their total weight is (m_ref/m)^kboost.
    mode "k": k_m^(-kboost) on every mode m >= 1 of every run, k_m = 2 pi m/L in
    1/sigma, no reference wavenumber (the paper's candidate form).
    kboost = 0 is the plain loss in every mode."""
    b = np.zeros(N // 2 + 1)
    if kboost <= 0:
        return b
    if mode == "k":
        k = 2 * np.pi * np.arange(1, N // 2 + 1) / L
        b[1:] = k ** (-kboost)
    elif mode == "all":
        m = np.arange(1, N // 2 + 1)
        b[1:] = (m_ref / m) ** kboost
    elif mode == "long12":
        if run.family in KBOOST_FAMILIES:
            for m in range(1, m_ref):
                b[m] = (m_ref / m) ** kboost - 1
    else:
        raise ValueError(f"kboost mode {mode}")
    return b


def make_batches(runs, sps, kboost=0.0, kboost_mode="long12"):
    """{conc: (Batch, [Run])} grouping runs by state point."""
    out = {}
    for c in sorted({r.conc for r in runs}):
        rs = [r for r in runs if r.conc == c]
        sp = sps[round(c, 4)]
        N = rs[0].n.shape[1]
        assert all(r.n.shape[1] == N for r in rs)
        g = geometry_of(sp, N)
        b = Batch(n=jnp.asarray(np.stack([r.n for r in rs])),
                  f=jnp.asarray(np.stack([r.f for r in rs])),
                  w=jnp.asarray(np.stack([r.w for r in rs])),
                  V=jnp.asarray(np.stack([r.V for r in rs])), geom=g,
                  kboost=jnp.asarray(np.stack([mode_boost(r, N, kboost, mode=kboost_mode, L=sp.L) for r in rs])))
        out[c] = (b, rs)
    return out


def geometries(sps, runs):
    """{conc: Geometry} for the state points present in `runs`."""
    return {c: b.geom for c, (b, _) in make_batches(runs, sps).items()}
