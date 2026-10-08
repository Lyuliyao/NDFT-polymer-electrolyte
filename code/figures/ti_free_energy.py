"""Free energy of switching on an external potential: the functionals against thermodynamic integration in MD (2026-10-06).
For the four held-out potentials of the untrained concentration c = 0.05 at both dielectric constants, MD runs at the
interior nodes of the five-point Gauss-Lobatto rule on lambda in [0, 1] (tools/make_ti_potentials.py) give
  dF_MD = int_0^1 <U_ext>_lambda dlambda,   <U_ext>_lambda = A sum_a int n_a^(lambda) V_a dz   (V zero mean: <U>_0 = 0).
Each functional gives the same quantity in closed form, dF = dF_id + dF_Coul + dF_ex + <U_ext>_1 at its Euler-Lagrange
profile, and by integrating its own <U_ext>_lambda (an integrable functional gives the same number by both routes).
The one-body network has no free energy; only its lambda integral is formed.  Units k_B T per ion pair.
    python ti_free_energy.py [--md]   -> si/ti_free_energy.json (+ prints); --md adds the MD integral once the runs exist"""
from figstyle import *
import glob, warnings
warnings.filterwarnings("ignore")
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.join(E2ROOT, "code"))
import learn.data as D
from learn.protocol import load_model
from learn.models import F_ex, ModelConfig, init_params
from learn.geometry import coulomb_phi
from learn.evaluate import el_solve
MD = "--md" in sys.argv
LOB = [(0.0, 1 / 20), ((1 - (3 / 7) ** 0.5) / 2, 49 / 180), (0.5, 16 / 45), ((1 + (3 / 7) ** 0.5) / 2, 49 / 180), (1.0, 1 / 20)]
GRID = sorted(set([x for x, _ in LOB] + list(np.linspace(0, 1, 21))))
TAGS = ("p01", "p07", "p11", "p13"); TI = {"p01": (70, 71, 72), "p07": (73, 74, 75), "p11": (76, 77, 78), "p13": (79, 80, 81)}
C = 0.05


def u_ext(n, V, g):
    return float(g.A * g.dz * np.sum(np.asarray(n) * np.asarray(V)))


def free_energy(params, cfg, n, V, g):
    """F_id + F_Coul + F_ex + int n V (total, k_B T), for an integrable model."""
    n = jnp.asarray(n); nb = jnp.full_like(n, g.nbar)
    fid = lambda x: float(g.A * g.dz * jnp.sum(x * (jnp.log(x) - 1.0)))
    fc = lambda x: float(0.5 * g.A * g.dz * jnp.sum((x[0] - x[1]) * coulomb_phi(x, g)))
    fex = lambda x: float(F_ex(params, cfg, x, g))
    return dict(id=fid(n) - fid(nb), coul=fc(n) - fc(nb), ex=fex(n) - fex(nb), ext=u_ext(n, V, g))


def simpson(x, y):
    from scipy.integrate import simpson as sp
    return float(sp(y, x=x))


out = {}
for eps in ("7.5", "2"):
    sy = system(eps); root = sy["root"]
    sps, runs = D.load_all(root=root)
    sp = sps[round(C, 4)]
    models = {"functional": sy["fun"], "pair closure": [sy["pair"]], "PB": [None]}
    c1 = sorted(glob.glob(os.path.join(root, "runs/learn", "joint_c1win_W30_*_kbK2_*_s[012]")))
    c1 = [d for d in c1 if "_f0" not in os.path.basename(d) and os.path.exists(os.path.join(d, "params.pkl"))]
    if c1: models["one-body network"] = c1
    for tag in TAGS:
        r = next(x for x in runs if abs(x.conc - C) < 1e-9 and x.tag == tag)
        N = r.n.shape[1]; g = D.geometry_of(sp, N); V = np.asarray(r.V); npair = g.n_pairs
        res = {"n_pairs": npair, "family": r.family, "kind": r.kind, "pp": [r.spec["pp_neutral_kT"], r.spec["pp_charged_kT"]]}
        # MD at lambda = 1 (existing run), block error
        def u_err(dd):
            """standard error of U_ext: from the stored blocks, else from the per-bin errors in quadrature (legacy files)"""
            if "cation_n_blocks" in dd.files:
                bb = np.stack([dd["cation_n_blocks"], dd["anion_n_blocks"]], axis=1)      # (B, 2, N)
                ub = np.array([u_ext(b, V, g) for b in bb]); return float(ub.std(ddof=1) / np.sqrt(len(ub)))
            er = np.stack([dd["cation_n_err"], dd["anion_n_err"]]); return float(g.A * g.dz * np.sqrt(np.sum((er * V) ** 2)))
        d = np.load(os.path.join(r.path, "profiles.npz"), allow_pickle=True)
        res["md_U1"] = [u_ext(r.n, V, g) / npair, u_err(d) / npair]
        if MD:
            us, es = [], []
            for lam_tag in TI[tag]:
                f = os.path.join(sp.path, f"field_p{lam_tag}", "profiles.npz")
                if not os.path.exists(f): us.append(None); es.append(None); continue
                dd = np.load(f, allow_pickle=True); lam = json.loads(str(dd["pot_json"]))["ti"]["lambda"]
                nn = np.stack([dd["cation_n"], dd["anion_n"]])
                us.append(u_ext(nn, V, g) / npair); es.append(u_err(dd) / npair)        # U_ext at lambda: profiles under lambda V, potential V
            if all(u is not None for u in us):
                vals = [0.0] + us + [res["md_U1"][0]]; errs = [0.0] + es + [res["md_U1"][1]]
                res["md_dF"] = [float(sum(w * v for (x, w), v in zip(LOB, vals))), float(np.sqrt(sum((w * e) ** 2 for (x, w), e in zip(LOB, errs))))]
                res["md_U_lambda"] = dict(zip([f"{x:.4f}" for x, _ in LOB], vals))
        for lab, dirs in models.items():
            rows = []
            for dd in dirs:
                cfg, ps = (ModelConfig("pb"), init_params(ModelConfig("pb"))) if dd is None else load_model(dd)
                U = {}
                n_prev = None
                for lam in GRID:
                    n_lam, info = el_solve(ps, cfg, g, lam * V, mixing=0.1, u0=None)
                    U[lam] = u_ext(n_lam, V, g) / npair
                    if abs(lam - 1.0) < 1e-12: n1, info1 = n_lam, info
                row = dict(U_lambda={f"{k:.4f}": v for k, v in U.items()}, converged=bool(info1["converged"]),
                           dF_TI=simpson(np.array(GRID), np.array([U[l] for l in GRID])),
                           dF_lobatto=float(sum(w * U[x] for x, w in LOB)))
                # second path (potentials with a neutral and a charged part): V_N first, then psi at full V_N
                VN = 0.5 * (V[0] + V[1]); PSI = 0.5 * (V[0] - V[1]); VNs = np.stack([VN, VN]); PSIs = np.stack([PSI, -PSI])
                if np.abs(VN).max() > 1e-9 and np.abs(PSI).max() > 1e-9:
                    leg1 = sum(w * u_ext(el_solve(ps, cfg, g, x * VNs, mixing=0.1)[0], VNs, g) for x, w in LOB) / npair
                    leg2 = sum(w * u_ext(el_solve(ps, cfg, g, VNs + x * PSIs, mixing=0.1)[0], PSIs, g) for x, w in LOB) / npair
                    row["dF_path2"] = float(leg1 + leg2)
                if cfg.variant != "c1win":
                    fe = free_energy(ps, cfg, n1, V, g); row["dF_closed"] = sum(fe.values()) / npair
                    row["parts"] = {k: v / npair for k, v in fe.items()}
                rows.append(row)
            res[lab] = rows
            cf = [x.get("dF_closed") for x in rows]; ti = [x["dF_TI"] for x in rows]; lo = [x["dF_lobatto"] for x in rows]
            p2 = [x.get("dF_path2") for x in rows]
            line = f"eps {eps} {tag} {lab:18s} dF closed {np.mean(cf) if cf[0] is not None else float('nan'):8.4f}  TI {np.mean(ti):8.4f}  lobatto {np.mean(lo):8.4f}"
            line += f"  path2 {np.mean(p2):8.4f}" if p2[0] is not None else ""
            line += f"  (U1 {np.mean([x['U_lambda']['1.0000'] for x in rows]):8.4f}; MD U1 {res['md_U1'][0]:8.4f} +- {res['md_U1'][1]:.4f})"
            line += f"  MD dF {res['md_dF'][0]:8.4f} +- {res['md_dF'][1]:.4f}" if "md_dF" in res else ""
            print(line, flush=True)
        out[f"{eps}/{tag}"] = res
json.dump(out, open(os.path.join(ROOT, "paper/figures/si/ti_free_energy.json"), "w"), indent=1, default=float)
print("wrote si/ti_free_energy.json")
