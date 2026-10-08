# Training specification: structure-preserving neural density functional (for Claude Code)

*2026-09-25. Data conventions: `cg-notes/03c_cgmd_setup.md`. Theory and proofs: `cg-notes/03b_sdft_learned_closure.md` §2–6, §10. Decisions taken on 2026-09-25: three functional variants are to be compared; radial basis fixed; Coulomb part analytic; depth of the nonlinear part is a hyperparameter, see §3.*

## 0. What is learned and from what

Learn $F^\theta_{\rm ex}[n_+,n_-]$ in
$$F=F_{\rm id}+F_{\rm Coul}+F^\theta_{\rm ex},\qquad \mu^{\rm int}_\alpha\equiv\frac{\delta(F_{\rm Coul}+F^\theta_{\rm ex})}{\delta n_\alpha}=e_\alpha\phi[n]+\mu^\theta_\alpha[n],$$
from planar MD profiles under static external potentials. Training identity (exact for the equilibrium functional):
$$f^{\rm int}_\alpha(z)=-n_\alpha(z)\,\partial_z\mu^{\rm int}_\alpha(z).$$
Left side is measured (dumped forces), right side is the model. Units: LJ, $k_BT=1$, lengths in $\sigma$, $e_\pm=\pm1$.

## 1. Data interface

Per accepted run (`status == "PASS"` in `acceptance.json`; `EXCLUDED` runs are skipped): from `profiles.npz` the grid `z` (bin centres, $\Delta z=0.1$), `n_plus, n_minus, f_plus, f_minus` (force densities, $z$-component), their per-bin standard errors `sig_n_*`, `sig_f_*`; from `potential.json` $V_N(z)$ and $\psi(z)$ on the same grid; from the state point's `zero_field/`: $L$, $N_\pm$, cross-section $A=L^2$, bulk $\bar n=N_\pm/L^3$. Gaussian-train runs are folded over their period before use. Held-out list from `splits/heldout.json` (fixed seed, never regenerated).

## 2. Fixed parts

**Radial basis (fixed).** $M$ normalized 3D Gaussians, $B_m(r)=(2\pi s_m^2)^{-3/2}e^{-r^2/2s_m^2}$, widths $s_m$ log-spaced in $[0.25,\,2.0]\,\sigma$, default $M=8$. Their planar reductions are 1D Gaussians of the same width, $\bar B_m(z)=(2\pi s_m^2)^{-1/2}e^{-z^2/2s_m^2}$ (this is exact: $2\pi\int_{|z|}^\infty rB_m(r)\,dr$). All convolutions are periodic in $z$ with period $L$ and done by FFT on the bin grid.

**Coulomb (analytic, never inside the network).** $\rho_Z=n_+-n_-$; in reduced units $\partial_z^2\phi=-4\pi l_B\rho_Z$ with $l_B=7.957$ ($\varepsilon_r=7.5$, $T^*=1$); solve in $k$-space, $\phi(k)=4\pi l_B\rho_Z(k)/k^2$ for $k\ne0$, $\phi(k{=}0)=0$ (electroneutral). Mean-field force density: $-n_\alpha e_\alpha\partial_z\phi$.

**Ideal part.** Not in the loss (it is the left side of YBG, $k_BT\partial_zn$); it enters only when solving the Euler–Lagrange equation for predictions (§6).

## 3. Model variants and architecture

Three variants, all trained and compared:

**V1, pair only.** $F^\theta_{\rm ex}=\tfrac12\sum_{\alpha\beta}\iint n_\alpha(\mathbf r)\,u_{\alpha\beta}(|\mathbf r-\mathbf r'|)\,n_\beta(\mathbf r')$, with $u_{\alpha\beta}(r)=\sum_mc_{\alpha\beta m}B_m(r)$, $c_{+-}=c_{-+}$: $3M$ parameters. Linear in the parameters, so the force-matching problem is linear least squares; solve it exactly. This is Avni's form with a learned short-range kernel and is the Gaussian (RPA) baseline of the family.

**V2, pair + local nonlinear.** V1 plus $\int d^3r\,\Phi_\theta(h^{(L)}(\mathbf r))$ with the deep nonlocal network below.

**V3, nonlinear only.** $\int d^3r\,\Phi_\theta(h^{(L)}(\mathbf r))$.

**Deep nonlocal network (depth $L\ge1$).** Alternate radial convolutions (nonlocal, linear, fixed kernels) with pointwise nonlinear maps (local, learned):

- Layer 0, lifting: $h^{(0)}_{m\alpha}(\mathbf r)=(B_m\star n_\alpha)(\mathbf r)$, $2M$ scalar fields. The learnable mixing of channels is the first linear layer of $\varphi^{(1)}$; this equals learning $w_{i\alpha}=\sum_mA_{i\alpha m}B_m$.
- Layer $\ell=1,\dots,L-1$, gating and re-convolution: $g^{(\ell)}_c(\mathbf r)=\varphi^{(\ell)}_c\big(h^{(\ell-1)}(\mathbf r),\,h^{(0)}(\mathbf r)\big)$, $c=1..C$ (pointwise MLP, SiLU, one hidden layer of width 64); $h^{(\ell)}_{mc}(\mathbf r)=(B_m\star g^{(\ell)}_c)(\mathbf r)$, $MC$ fields. The skip input $h^{(0)}$ keeps the raw weighted densities available at every depth.
- Readout: $F^\theta_{\rm ex}=\int d^3r\,\Phi_\theta\big(h^{(L-1)}(\mathbf r),\,h^{(0)}(\mathbf r)\big)$, $\Phi_\theta$ a pointwise MLP (two hidden layers of 64, SiLU) to a scalar. $L=1$ means readout directly on $h^{(0)}$, which is the one-layer $\int\Phi(\bar n)$ of the note.
- Optional (flag `grad_invariants`): also feed $\Phi_\theta$ the parity-even products $\partial_zh_i\,\partial_zh_j$ of a few channels (the field analogue of Gram-matrix invariants $\mathbf K\mathbf K^{\top}$). Off by default; try after the depth scan.

Depth is a hyperparameter: scan $L\in\{1,2,3\}$, $C\in\{4,8\}$. The receptive field grows as roughly $L\times3s_{\max}$; keep it below about $6\,\sigma$ so that the learned part stays short-ranged and the long range stays analytic.

**Why the properties survive depth.** Every $h^{(\ell)}$ is a scalar field obtained from scalar fields by radial convolution and pointwise maps, so under any isometry it transforms as a scalar field; the final $F$ is a scalar; therefore the proofs of E(3) covariance, integrability, the Noether sum rules and the Stillinger–Lovett limit in 03b §3–6 go through unchanged (they use only radial kernels, pointwise $C^2$ maps and invariance of a scalar). The force is $-n_\alpha\partial_z(\delta F/\delta n_\alpha)$ by automatic differentiation of the scalar; do not learn a force network.

**Correspondence with the particle-based symmetry-preserving mean-field network** (for orientation only): particle positions $\mathbf X_i$ and empirical measure $\mu$ → density fields $n_\alpha$; centering → translation invariance built into convolutions; radial maps $v_\ell(r)$ → radial kernels $B_m$ and pointwise $\varphi^{(\ell)}$; weighted centering with $\exp(-V^{(\ell-1)})$ → gated fields $g^{(\ell)}=\varphi^{(\ell)}(h^{(\ell-1)})$ that are re-convolved; Gram invariants $\mathbf K\mathbf K^{\top}$ → gradient invariants $\partial_zh_i\partial_zh_j$; equivariant readout $Q\mathbf K$ → the conservative force $-n\nabla\delta F/\delta n$ from the scalar readout.

## 4. Loss

$$\mathcal L(\theta)=\sum_{\rm runs}\sum_\alpha\sum_{\rm bins}\frac{\big[f^{\rm int,MD}_\alpha+n_\alpha\,\partial_z\big(e_\alpha\phi[n]+\mu^\theta_\alpha[n]\big)\big]^2}{\sigma_{f,\alpha}^2+\epsilon^2}$$
with the MD densities as input, per-bin inverse-variance weights from the acceptance error bars, and a floor $\epsilon$ equal to the median $\sigma_f$ of the run. Regularization: weight decay $10^{-5}$; stability penalty $\lambda\sum_{\rm state\ points}\sum_k{\rm relu}\big(-\lambda_{\min}[\bar N^{-1}+W(k)]\big)$ with $W(k)$ the bulk Hessian of $F^\theta_{\rm ex}$ at the uniform density $\bar n$ of each state point, obtained by a Jacobian-vector product of the map $n\mapsto\mu^\theta$ on a single-mode cosine perturbation, $k$ on the grid up to $2\pi/\Delta z$. Optimizer Adam, lr $10^{-3}$ with cosine decay, full-batch (the data are small), early stopping on the held-out loss.

## 5. Protocol

1. V1 by linear least squares on $c=0.04$ training runs; report held-out force residual and $S(k)$; this is the first baseline of the learned family.
2. Learning curve on $c=0.04$ for V3 with $L=1$: train on 4, 8, 12, 20 runs, evaluate on the 4 held-out; then depth scan $L\in\{1,2,3\}$ on all 20.
3. V2 and V3 at the chosen depth; joint training on $c=0.01,0.02,0.04$ with one shared $\theta$; then the transfer tests of §6.
4. Add $c=0.06$, $0.08$ when accepted, retrain with the same architecture; $c=0.05$ stays entirely held out.

## 6. Evaluation (every trained model reports all of these)

- Held-out force residual: $\chi^2$ per bin of the loss on held-out runs.
- Held-out profile prediction: solve the canonical Euler–Lagrange equation self-consistently, $n_\alpha(z)=N_\alpha\,e^{-[V_\alpha+e_\alpha\phi+\mu^\theta_\alpha]}/\int e^{-[\dots]}$ (Picard iteration, mixing 0.05–0.2, converge to $10^{-6}$), and compare $n_\alpha(z)$ with MD: relative $L^2$ error and error at the density maximum and minimum.
- Zero-field structure: $S(k)=[\bar N^{-1}+\tilde v^{\rm Coul}(k)+W(k)]^{-1}$ against $S_{\alpha\beta}(k)$ from the zero-field run at each concentration ($k_BT=1$).
- Thermodynamic factor $\Gamma=1+\tfrac{\bar n}{2}\sum_{\alpha\beta}W_{\alpha\beta}(0)$ per concentration.
- Stillinger–Lovett extrapolation $A$ of the model's $S_{ZZ}$ (should be exactly 1) as a code check.
- Transfer: train on $\{0.01,0.04\}$, predict $0.02$; train on $\{0.02,0.04\}$, predict $0.01$; later predict $0.05$ and the zero-field $S(k)$ at $0.12$.
- Baselines on the same held-out runs, all solved with the normalization constraints: Poisson–Boltzmann ($F^\theta_{\rm ex}=0$), Avni's cut-off kernel ($u=-\theta(a-r)\,l_B/r$ with $a=1$), and V1.

## 7. Implementation notes

JAX. One-dimensional periodic grid of 245 points per state point (different $L$ per concentration: pad or handle per state point). Convolutions by `jnp.fft`. $\mu^\theta_\alpha=\delta F^\theta_{\rm ex}/\delta n_\alpha$ by `jax.grad` of the discretized $F$ with respect to the density arrays, divided by $A\,\Delta z$. Hessian-vector products by `jax.jvp` of the $\mu$-map. Check on synthetic data first: (i) generate profiles from a known V1 kernel with the Euler–Lagrange solver, recover $c_{\alpha\beta m}$ by least squares; (ii) verify $\delta F/\delta n$ against finite differences; (iii) verify that the model's $S_{ZZ}$ obeys Stillinger–Lovett to $10^{-6}$ after extrapolation.
