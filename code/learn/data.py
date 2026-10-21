import csv
import glob
import json
import os
import warnings
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from .geometry import Geometry, SPECIES, make_geometry, fold_periodic

ROOT = os.environ.get("SIP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT_SEED = 20260925
N_HELDOUT = 4


@dataclass
class StatePoint:
    conc: float
    path: str
    zero_field: str
    L: float
    n_pairs: int
    lB: float
    volume: float
    box: str = ""
    eps_r: str = ""
    eps_index: int = 0

    @property
    def sk(self):
        f = os.path.join(self.zero_field, "Sk.npz")
        return dict(np.load(f)) if os.path.exists(f) else None


def discover_state_points(root=ROOT):

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


BOX_KEY_OFFSET = 1000.0


EPS_KEY_OFFSET = 10.0


def state_key(conc, box_index=0, eps_index=0):
    return round(float(conc) + BOX_KEY_OFFSET * box_index + EPS_KEY_OFFSET * eps_index, 4)


def extra_state_points():

    out = []
    spec = os.environ.get("SIP_EXTRA_STATES", "")
    for i, item in enumerate([x for x in spec.split(":") if x], start=1):
        label, d = item.split("=", 1)
        d = d.rstrip("/")
        zf = os.path.join(d, "zero_field")
        dj = json.load(open(os.path.join(zf, "D.json")))
        skf = os.path.join(zf, "Sk.npz")
        sp = StatePoint(conc=float(dj["conc"]), path=d, zero_field=zf, L=float(dj["lz"]), n_pairs=int(dj["n_pairs"]),
                        lB=float(np.load(skf)["lB"]) if os.path.exists(skf) else 7.957187947537492,
                        volume=float(dj["volume"]), box=label)
        out.append((i, state_key(sp.conc, i), sp))
    return out


EXTRA_STATUS = set(os.environ.get("SIP_EXTRA_STATUS", "PASS").split())


def eps_roots():

    spec = os.environ.get("SIP_EPS_ROOTS", "")
    return [(i, lab, d.rstrip("/")) for i, (lab, d) in enumerate((x.split("=", 1) for x in spec.split(":") if x), start=1)]


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


@dataclass
class Run:
    conc: float
    tag: str
    family: str
    kind: str
    path: str
    status: str
    fold_q: int
    n: np.ndarray
    sig_n: np.ndarray
    f: np.ndarray
    sig_f: np.ndarray
    V: np.ndarray
    w: np.ndarray
    eps: np.ndarray
    spec: dict = field(repr=False)
    uncertainty_method: str = "legacy_diagonal"
    box: str = ""
    box_index: int = 0
    eps_r: str = ""
    eps_index: int = 0

    @property
    def key(self):
        k = f"c{self.conc:g}/{self.tag}" if not self.box else f"c{self.conc:g}/{self.box}/{self.tag}"
        return f"e{self.eps_r}/{k}" if self.eps_r else k

    @property
    def state_key(self):

        return state_key(self.conc, self.box_index, self.eps_index)

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

    return tuple(s for s in SPECIES if f"{s}_n" in d)


def n_species(runs):
    return runs[0].n.shape[0] if runs else 2


def profile_uncertainties(d, q=1, mode="recorded"):

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


def load_run(sp, tag, verdicts=None, excluded=None, fold=True, uncertainty="recorded", box_index=0):
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
               spec=spec, uncertainty_method=uncertainty_method, box=sp.box, box_index=box_index,
               eps_r=sp.eps_r, eps_index=sp.eps_index)


def load_all(root=ROOT, concs=None, fold=True, uncertainty="recorded"):

    sps, runs = {}, []
    for i, lab, rt in eps_roots() or [(0, "", root)]:
        verd, excl = load_verdicts(rt), load_excluded(rt)
        for c, sp in sorted(discover_state_points(rt).items()):
            sp = replace(sp, eps_r=lab, eps_index=i)
            sps[state_key(c, 0, i)] = sp
            if concs is not None and not any(abs(c - cc) < 1e-6 for cc in concs):
                continue
            for p in sorted(glob.glob(os.path.join(sp.path, "field_p*"))):
                r = load_run(sp, os.path.basename(p)[6:], verd, excl, fold=fold,
                             uncertainty=uncertainty)
                if r is not None:
                    runs.append(r)
    extra_tags = os.environ.get("SIP_EXTRA_TAGS", "").split()
    for i, key, sp in extra_state_points():
        sps[key] = sp
        everd = load_verdicts(os.path.dirname(os.path.dirname(sp.path)))
        for p in sorted(glob.glob(os.path.join(sp.path, "field_p*"))):
            tag = os.path.basename(p)[6:]
            if extra_tags and tag not in extra_tags:
                continue
            r = load_run(sp, tag, everd, set(), fold=fold, uncertainty=uncertainty, box_index=i)
            if r is not None:
                runs.append(r)
    return sps, runs


def geometry_of(sp, N):
    return make_geometry(sp.L, N, sp.n_pairs, sp.lB, sp.conc)


def _strata(r):
    if r.family == "gauss":
        return "gauss"
    if r.family == "mixed":
        return "mixed"
    return "fourier_both" if r.kind == "both" else "fourier_one"


def heldout_tags(runs, root=ROOT, regenerate=False, seed=SPLIT_SEED, n_heldout=N_HELDOUT):

    roots = eps_roots()
    if roots:
        return {lab: _heldout_tags_root([r for r in runs if r.eps_r == lab], rt, regenerate, seed, n_heldout)
                for _, lab, rt in roots}
    return _heldout_tags_root(runs, root, regenerate, seed, n_heldout)


def _heldout_tags_root(runs, root, regenerate, seed, n_heldout):
    f = os.path.join(root, "splits", "heldout.json")
    out = {}
    if os.path.exists(f) and not regenerate:
        out = {float(k): v for k, v in json.load(open(f)).items()}
    runs = [r for r in runs if not r.box]
    concs = [c for c in sorted({r.conc for r in runs}) if c not in out
             and any(r.conc == c and r.status == "PASS" for r in runs)]
    if not concs:
        return out
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

    val, rest = [], []
    for e, c in sorted({(r.eps_r, r.conc) for r in train_runs}):
        rest += [r for r in train_runs if (r.eps_r, r.conc) == (e, c) and (r.family == "linear" or r.box)]
        rs = [r for r in train_runs if (r.eps_r, r.conc) == (e, c) and r.family != "linear" and not r.box]
        order = np.random.default_rng([seed, 7, int(round(c * 10000))]).permutation(len(rs))
        pick = set(order[:nval].tolist())
        val += [rs[i] for i in sorted(pick)]
        rest += [rs[i] for i in range(len(rs)) if i not in pick]
    return val, rest


def split_runs(runs, heldout):
    tr, ho = [], []
    for r in runs:
        if r.box:
            if r.status in EXTRA_STATUS:
                tr.append(r)
            continue
        if r.status != "PASS":
            continue
        ho_c = heldout.get(r.eps_r, {}) if r.eps_r else heldout
        (ho if r.tag in ho_c.get(r.conc, []) else tr).append(r)
    return tr, ho


class Batch(NamedTuple):

    n: jnp.ndarray
    f: jnp.ndarray
    w: jnp.ndarray
    V: jnp.ndarray
    geom: Geometry
    kboost: jnp.ndarray


KBOOST_FAMILIES = ("long", "long3")


def mode_boost(run, N, kboost=0.0, m_ref=3, mode="long12", L=None):

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

    out = {}
    for c in sorted({r.state_key for r in runs}):
        rs = [r for r in runs if r.state_key == c]
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

    return {c: b.geom for c, (b, _) in make_batches(runs, sps).items()}
