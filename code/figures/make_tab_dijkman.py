"""SI Table (section S9.5): the paper's functional and the convolutional free energy of Dijkman et al., each trained
by force matching and by pair-correlation matching, both eps_r.  From <eps2 root>/runs/learn/interpretation/dijkman_summary.json
(figs/dijkman_summary.py).  Writes paper/output/latex/si/si_tab_dijkman.tex."""
import json, numpy as np
d = json.load(open("/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2/runs/learn/interpretation/dijkman_summary.json"))
ROWS = [("ours, force", "neural functional", "force matching"), ("ours, pcm", "neural functional", "pair-correlation matching"),
        ("CNN, force", "convolutional", "force matching"), ("CNN, force, 20k steps", "convolutional", "force matching, 20{,}000 steps"),
        ("CNN, force, shifted and mirrored copies", "convolutional", "force matching, augmented$^{*}$"),
        ("CNN, pcm", "convolutional", "pair-correlation matching")]
def ndig(x):
    """decimals for a chi2-like number: 2 below 1, 1 below 10, 0 below 1000, -2 (hundreds) above"""
    return 2 if x < 1 else 1 if x < 10 else 0 if x < 1000 else -2
def num(x, d=None):
    d = ndig(x) if d is None else d
    return f"{round(x, d):.0f}" if d <= 0 else f"{x:.{d}f}"
def f(v, fmt):
    v = [x for x in v if x == x]
    if not v: return "--"
    m, sd = np.mean(v), (np.std(v, ddof=1) if len(v) > 1 else None)
    if fmt == "chi2":
        d = ndig(m)
        return num(m, d) + (f"$\\pm${num(sd, d)}" if sd is not None else "")
    d = 2 if (sd is not None and sd < 0.05) else 1
    return f"{m:.{d}f}" + (f"$\\pm${sd:.{d}f}" if sd is not None else "")
def pct(x): return f"{x:.0f}" if x >= 10 else f"{x:.1f}"
def sci(x):
    if x < 1e-2:
        e = int(np.floor(np.log10(x))); m = x / 10 ** e
        return f"${m:.0f}\\times10^{{{e}}}$" if round(m) != 1 else f"$10^{{{e}}}$"
    return f"{x:.2f}" if x < 1 else f"{x:.1f}"
lines = []
for e in ("7.5", "2"):
    for key, form, sig in ROWS:
        ss = d.get(e, {}).get(key)
        if not ss: continue
        noe = np.concatenate([s["noether_all"] for s in ss]); refl = np.concatenate([s["refl_all"] for s in ss])
        lines.append(f"{e if key == 'ours, force' else ''} & {form} & {sig} & {f([s['ho_chi2'] for s in ss], 'chi2')} & {f([s['ho_L2'] for s in ss], '%.1f')} & "
                     f"{f([s['c5_chi2'] for s in ss], 'chi2')} & {sum(s['el'] for s in ss)}/{sum(s['nel'] for s in ss)} & "
                     f"{pct(np.mean([s['Sk_ZZ'] for s in ss]))} / {pct(np.mean([s['Sk_NN'] for s in ss]))} & {max(np.max(np.abs(s['Gamma_dev'])) for s in ss):.1f} & "
                     f"{sci(noe.max())} ({sci(np.mean(noe))}) & {sci(refl.max())}\\\\")
tab = r"""\begin{table}[h]
\centering
\caption{Architecture and training signal: the neural functional and the convolutional free energy of ref.~\cite{Dijkman2025}, each trained by force matching and by pair-correlation matching on the zero-field structure factors (three seeds; one seed for the rows with 20{,}000 steps and with augmented data). Force residual $\chi^2$ and profile error on the held-out runs (without p27 at $\varepsilon_r=2$), $\chi^2$ at the untrained $c=0.05$, converged Euler--Lagrange solutions on the held-out and $c=0.05$ runs, the largest structure-factor error at $c=0.02$ to 0.08 ($S_{ZZ}$ / $S_{NN}$), the largest deviation of $\Gamma(k_1)$ from MD in MD standard errors, the net internal force relative to the total (maximum over runs and seeds; arithmetic mean over runs and seeds in parentheses) and the reflection error.}
\label{tab:S:dijkman}
\begin{tabular}{lllcccccccc}
$\varepsilon_r$ & form & signal & held-out $\chi^2$ & profile (\%) & $c=0.05$: $\chi^2$ & EL & $S(k)$ (\%) & $|\Delta\Gamma|$ & net force & reflection\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}\par\smallskip
\addtabletext{Profile errors of the convolutional free energy include Euler--Lagrange iterations that did not converge. $^{*}$Seven random shifts and the mirror image of every copy of each training run (16 copies).}
\end{table}
"""
open("/mnt/gs21/scratch/lyuliyao/salt_in_polymer/paper/output/latex/si/si_tab_dijkman.tex", "w").write(tab)
print(tab)
