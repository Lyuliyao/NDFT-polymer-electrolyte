# A structure-preserving learned closure for the SDFT of salt-doped polymers

*Working note, 2026-09-20. Builds on Avni, Adar, Andelman & Orland, PRL 128, 098002 (2022), arXiv:2110.07008.*
*(Copied into the paper folder on 2026-09-26 from the user's message; this is the theory source of truth for the Theory section. Numbering of propositions is referenced from `notes/result_summary.md` Claim 7 and `output/doc/manuscript_outline.md`.)*

## 0. Purpose

The stochastic density functional theory (SDFT) of Avni et al. obtains the conductivity of an aqueous 1:1 electrolyte from one modelling input, the force density $\mathbf f_\alpha$ acting on each ionic species. In a salt-doped polymer that input can no longer be a cut-off Coulomb force: the ions are solvated by the chain, the dielectric constant is low, and ion pairing is strong. We keep the long-range Coulomb part analytic and learn the remainder from data, not as a vector field but as a scalar free-energy functional.

This note fixes the notation, states exactly what is assumed about the learned functional, and proves that four physical properties then hold by construction: E(3) covariance (Section 3); integrability, and with it a Boltzmann equilibrium and the fluctuation–dissipation relations (Section 4); the Noether sum rules together with an explicit symmetric stress tensor (Section 5); and the exact long-wavelength electrostatics, namely electroneutrality, the Stillinger–Lovett condition and the Debye–Hückel–Onsager limiting law (Section 6). Section 7 lists what is *not* guaranteed, Section 8 reports a numerical check of every proposition, Section 9 collects remarks specific to polymers, and Section 10 proposes a statics-first training scheme, in which the model learns the response of the ion densities to external fields from MD or CG-MD data, and lists the quantities that are then genuine predictions. The changes made to the earlier draft are listed at the end.

## 1. Model

### 1.1 Dynamics

Let $n_\alpha(\mathbf r,t)$, $\alpha\in\{+,-\}$, be the number densities of cations and anions, with charges $e_\alpha$, mobilities $\mu_\alpha$ and diffusion coefficients $D_\alpha=\mu_\alpha k_BT$. The extension to more species is purely notational. The SDFT is the conservation law

$$\partial_t n_\alpha=-\nabla\cdot\mathbf j_\alpha, \tag{1.1}$$

$$\mathbf j_\alpha=n_\alpha\mathbf u-D_\alpha\nabla n_\alpha+\mu_\alpha\mathbf f_\alpha-\sqrt{2D_\alpha n_\alpha}\,\boldsymbol\zeta_\alpha, \tag{1.2}$$

where $\boldsymbol\zeta_\alpha$ is Gaussian white noise, $\langle\zeta^p_\alpha(\mathbf r,t)\zeta^q_\beta(\mathbf r',t')\rangle=\delta_{\alpha\beta}\delta_{pq}\delta(\mathbf r-\mathbf r')\delta(t-t')$, interpreted in the Itô sense ($p,q$ are Cartesian indices). The velocity $\mathbf u$ of the medium is incompressible and obeys the Stokes force balance

$$\nabla\cdot\mathbf u=0,\qquad \eta\nabla^2\mathbf u-\nabla p+\sum_\alpha\mathbf f_\alpha=0. \tag{1.3}$$

The same $\mathbf f_\alpha$ enters (1.2) and (1.3). This is action and reaction between the ions and the medium, and any learned model has to keep it. Because the network only ever enters through the flux $\mathbf j_\alpha$, the particle number of each species and the total charge are conserved exactly, whatever the network outputs.

### 1.2 Force density

The force density has an external part, due to the applied field $\mathbf E_0$, and an internal part, due to the interactions among the ions and between the ions and the medium they are dissolved in:

$$\mathbf f_\alpha=n_\alpha e_\alpha\mathbf E_0+\mathbf f^{\rm int}_\alpha,\qquad \mathbf f^{\rm int}_\alpha=-n_\alpha\nabla\mu^{\rm int}_\alpha,\qquad \mu^{\rm int}_\alpha\equiv\frac{\delta F_{\rm int}}{\delta n_\alpha},\qquad F_{\rm int}=F_{\rm Coul}+F^\theta_{\rm ex}. \tag{1.4}$$

The uniform field is kept outside the free energy on purpose. In a periodic system $\mathbf E_0$ is not the gradient of a periodic potential; it is a non-conservative drive, and that is why a steady current exists. (The earlier draft called $F_{\rm int}$ "$F_{\rm ex}$". The name is changed so that "ex" is reserved for the learned remainder.)

With the ideal-gas free energy

$$F_{\rm id}[n]=k_BT\sum_\alpha\int d^3r\;n_\alpha\big[\ln(n_\alpha\Lambda_\alpha^3)-1\big],\qquad F\equiv F_{\rm id}+F_{\rm int},$$

the diffusion term is $-D_\alpha\nabla n_\alpha=-\mu_\alpha n_\alpha\nabla(\delta F_{\rm id}/\delta n_\alpha)$, and the flux takes the gradient-flow form

$$\mathbf j_\alpha=n_\alpha\mathbf u-\mu_\alpha n_\alpha\nabla\frac{\delta F}{\delta n_\alpha}+\mu_\alpha n_\alpha e_\alpha\mathbf E_0-\sqrt{2k_BT\mu_\alpha n_\alpha}\,\boldsymbol\zeta_\alpha. \tag{1.5}$$

Every structural statement below is a statement about $F$.

### 1.3 Coulomb part

With the charge density $\rho_Z(\mathbf r)=\sum_\alpha e_\alpha n_\alpha(\mathbf r)$,

$$F_{\rm Coul}[n]=\frac12\iint d^3r\,d^3r'\,\frac{\rho_Z(\mathbf r)\rho_Z(\mathbf r')}{4\pi\varepsilon_0\varepsilon|\mathbf r-\mathbf r'|},\qquad \frac{\delta F_{\rm Coul}}{\delta n_\alpha}=e_\alpha\phi,\qquad -\varepsilon_0\varepsilon\nabla^2\phi=\rho_Z. \tag{1.6}$$

In Fourier space the Coulomb interaction matrix is $\tilde v^{\rm Coul}_{\alpha\beta}(k)=e_\alpha e_\beta/(\varepsilon_0\varepsilon k^2)$, a rank-one positive semi-definite matrix. Here $\varepsilon$ is the constant dielectric constant of the salt-free medium; its concentration dependence and all other short-range physics are left to $F^\theta_{\rm ex}$. A variant that is numerically more convenient keeps only the smooth part ${\rm erf}(r/\sigma)/r$ in $F_{\rm Coul}$, so that $\tilde v^{\rm Coul}$ is multiplied by $e^{-k^2\sigma^2/4}$, and moves ${\rm erfc}(r/\sigma)/r$ into the learned part. The difference between the two kernels is regular at $k=0$, so nothing in Section 6 changes.

## 2. The learned functional

### 2.1 Ansatz and assumptions

Define $K$ weighted densities as convolutions of the species densities with radial weight functions,

$$\bar n_i(\mathbf r)=\sum_\alpha\,(w_{i\alpha}\star n_\alpha)(\mathbf r)=\sum_\alpha\int d^3r'\;w_{i\alpha}(|\mathbf r-\mathbf r'|)\,n_\alpha(\mathbf r'),\qquad i=1,\dots,K, \tag{2.1}$$

and let the learned free energy be a local function of them,

$$F^\theta_{\rm ex}[n]=\int d^3r\;\Phi_\theta\big(\bar n_1(\mathbf r),\dots,\bar n_K(\mathbf r)\big). \tag{2.2}$$

The earlier draft wrote $\bar n^{(i)}=w_i(|\mathbf r|)\,n_\alpha$. The weighted density has to be a convolution, and it needs the species index; (2.1) is the corrected definition. A per-species channel is the special case in which $w_{i\alpha}$ vanishes for all but one $\alpha$. On an unbounded domain with a uniform background one integrates $\Phi_\theta(\bar n)-\Phi_\theta(\bar n^{\rm b})$ instead, which changes no derivative.

The assumptions used in the proofs are the following.

(A1) *Radial weights.* Each $w_{i\alpha}$ depends on $|\mathbf r|$ only. In practice it is a learnable radial function, for example an expansion in radial basis functions with a smooth cut-off $r_c$.

(A2) *Short range.* Each $w_{i\alpha}$ is integrable and has a finite second moment, $\int d^3r\,r^2|w_{i\alpha}(r)|<\infty$. A cut-off guarantees this. Then $\tilde w_{i\alpha}(k)$ is real, depends on $|\mathbf k|$ only, and $\tilde w_{i\alpha}(k)=\tilde w_{i\alpha}(0)+O(k^2)$.

(A3) *Smooth scalar energy density.* $\Phi_\theta:\mathbb R^K\to\mathbb R$ is twice continuously differentiable (smooth activations such as SiLU, softplus or tanh, not ReLU) and has no explicit dependence on position or direction. It may depend on temperature and on the parameters $\theta$.

(A4) *Admissible densities.* Densities live either in a periodic box or on $\mathbb R^3$ as localized perturbations of a uniform background, so that all integrals converge and boundary terms vanish.

### 2.2 First and second functional derivatives

Write $\Phi_i(\mathbf r)\equiv\partial\Phi_\theta/\partial\bar n_i$ and $\Phi_{ij}(\mathbf r)\equiv\partial^2\Phi_\theta/\partial\bar n_i\partial\bar n_j$, both evaluated at $\bar n(\mathbf r)$. Since a radial kernel is its own adjoint under convolution,

$$\mu^\theta_\alpha(\mathbf r)\equiv\frac{\delta F^\theta_{\rm ex}}{\delta n_\alpha(\mathbf r)}=\sum_i\,(w_{i\alpha}\star\Phi_i)(\mathbf r), \tag{2.3}$$

$$\frac{\delta^2F^\theta_{\rm ex}}{\delta n_\alpha(\mathbf r)\,\delta n_\beta(\mathbf r')}=\sum_{ij}\int d^3r''\;w_{i\alpha}(|\mathbf r-\mathbf r''|)\,\Phi_{ij}(\mathbf r'')\,w_{j\beta}(|\mathbf r''-\mathbf r'|). \tag{2.4}$$

In a uniform bulk state $\Phi_{ij}=H_{ij}$ is a constant symmetric matrix, the kernel (2.4) depends on $\mathbf r-\mathbf r'$ only, and its Fourier transform is

$$W_{\alpha\beta}(k)=\sum_{ij}\tilde w_{i\alpha}(k)\,H_{ij}\,\tilde w_{j\beta}(k)=\big[\tilde w(k)^{\!\top}H\,\tilde w(k)\big]_{\alpha\beta},\qquad c^{\rm SR}_{\alpha\beta}(k)=-\beta W_{\alpha\beta}(k). \tag{2.5}$$

Here $c^{\rm SR}$ is the short-range part of the direct correlation function. The full direct correlation function is $c_{\alpha\beta}(k)=-\beta e_\alpha e_\beta/(\varepsilon_0\varepsilon k^2)+c^{\rm SR}_{\alpha\beta}(k)$. In an implementation (2.3) is produced by reverse-mode automatic differentiation of (2.2); the explicit formulas are needed only for the proofs.

### 2.3 Optional tensorial channels

Vector and tensor weighted densities, $\bar{\mathbf n}^{(1)}_i=\sum_\alpha\int w_{i\alpha}(|\mathbf s|)\,\hat{\mathbf s}\,n_\alpha(\mathbf r')\,d^3r'$ and $\bar{\mathsf n}^{(2)}_i=\sum_\alpha\int w_{i\alpha}(|\mathbf s|)\,(\hat{\mathbf s}\hat{\mathbf s}-\mathsf I/3)\,n_\alpha(\mathbf r')\,d^3r'$ with $\mathbf s=\mathbf r-\mathbf r'$, may be added as long as $\Phi_\theta$ receives only their rotational invariants ($|\bar{\mathbf n}^{(1)}_i|^2$, $\bar{\mathbf n}^{(1)}_i\!\cdot\bar{\mathbf n}^{(1)}_j$, $\bar{\mathbf n}^{(1)}_i\!\cdot\bar{\mathsf n}^{(2)}_j\!\cdot\bar{\mathbf n}^{(1)}_k$, ${\rm tr}\,(\bar{\mathsf n}^{(2)}_i)^m$, and so on). Propositions 1, 2 and 3(a,b) use only the invariance of $F$ and the fact that the force is a functional gradient, so they carry over unchanged. The explicit stress of Proposition 3(c) and formula (2.5) are written for scalar channels.

## 3. Proposition 1: translation invariance and rotation covariance

Let $g=(R,\mathbf a)$ with $R\in O(3)$ act on points by $g\mathbf r=R\mathbf r+\mathbf a$ and on densities by $(T_gn)_\alpha(\mathbf r)=n_\alpha(g^{-1}\mathbf r)$.

**Proposition 1.** Under (A1), (A3), (A4), for every $g\in E(3)$ and every admissible $n$:

(i) $F_{\rm int}[T_gn]=F_{\rm int}[n]$;

(ii) the chemical potential is a scalar field, $\mu^{\rm int}_\alpha[T_gn](g\mathbf r)=\mu^{\rm int}_\alpha[n](\mathbf r)$;

(iii) the internal force density is a vector field, $\mathbf f^{\rm int}_\alpha[T_gn](g\mathbf r)=R\,\mathbf f^{\rm int}_\alpha[n](\mathbf r)$;

(iv) with the field treated as an input vector, $\mathbf f_\alpha[T_gn;R\mathbf E_0](g\mathbf r)=R\,\mathbf f_\alpha[n;\mathbf E_0](\mathbf r)$.

*Proof.* Step 1, the weighted densities are scalar fields. Substitute $\mathbf r'=g\mathbf s$ in (2.1); the Jacobian is $|\det R|=1$ and $|g\mathbf x-g\mathbf s|=|\mathbf x-\mathbf s|$ because $g$ is an isometry, so

$$\bar n_i[T_gn](\mathbf r)=\sum_\alpha\int d^3s\;w_{i\alpha}\big(|g^{-1}\mathbf r-\mathbf s|\big)\,n_\alpha(\mathbf s)=\bar n_i[n](g^{-1}\mathbf r).$$

Step 2, invariance of $F$. By Step 1 and the substitution $\mathbf r=g\mathbf s$,

$$F^\theta_{\rm ex}[T_gn]=\int d^3r\;\Phi_\theta\big(\bar n[n](g^{-1}\mathbf r)\big)=\int d^3s\;\Phi_\theta\big(\bar n[n](\mathbf s)\big)=F^\theta_{\rm ex}[n].$$

It matters here that $\Phi_\theta$ has no explicit dependence on position or direction (A3). The same two substitutions in (1.6) give $F_{\rm Coul}[T_gn]=F_{\rm Coul}[n]$, since the kernel depends on $|\mathbf r-\mathbf r'|$ only. This proves (i).

Step 3, an invariant functional has a covariant derivative. Apply (i) to $n+\epsilon h$ and differentiate at $\epsilon=0$; $T_g$ is linear, so

$$\sum_\alpha\int d^3r\;\mu^{\rm int}_\alpha[T_gn](\mathbf r)\,h_\alpha(g^{-1}\mathbf r)=\sum_\alpha\int d^3s\;\mu^{\rm int}_\alpha[n](\mathbf s)\,h_\alpha(\mathbf s).$$

Substituting $\mathbf r=g\mathbf s$ on the left and using that $h$ is arbitrary gives (ii).

Step 4, the gradient of a scalar field is a vector field. By (ii), $\mu^{\rm int}_\alpha[T_gn](\mathbf r)=\mu^{\rm int}_\alpha[n](R^{\top}(\mathbf r-\mathbf a))$, and the chain rule gives $\nabla_{\mathbf r}\,\mu^{\rm int}_\alpha[T_gn](\mathbf r)=R\,(\nabla\mu^{\rm int}_\alpha[n])(g^{-1}\mathbf r)$. Multiplying by $-(T_gn)_\alpha(\mathbf r)=-n_\alpha(g^{-1}\mathbf r)$ and evaluating at $g\mathbf r$ gives (iii). The external term $n_\alpha e_\alpha\mathbf E_0$ transforms in the same way when $\mathbf E_0\to R\mathbf E_0$, which gives (iv). $\blacksquare$

*Remarks.* The proof covers improper rotations, so parity is included. Since gradient, divergence and Laplacian commute with isometries and the law of the white noise is invariant, (iv) implies that the law of the solution of (1.1)–(1.3) is E(3)-covariant: if $(n,\mathbf u,p)$ solves the equations with field $\mathbf E_0$, then $(T_gn,\;R\,\mathbf u\circ g^{-1},\;p\circ g^{-1})$ solves them with field $R\mathbf E_0$. In a periodic box only translations and the point group of the lattice are exact symmetries; continuous rotations then hold up to finite-size and discretization error, which is at round-off level for well-resolved smooth fields (Section 8, T1).

Note what was *not* required: no equivariant network architecture. Equivariance of the force follows from the invariance of one scalar, which is how energy-based machine-learned interatomic potentials obtain equivariant forces.

## 4. Proposition 2: integrability, Boltzmann equilibrium, fluctuation–dissipation

### 4.1 Symmetry of the direct correlation functions

**Proposition 2(a).** Under (A1) and (A3), for every admissible $n$,

$$\frac{\delta\mu^{\rm int}_\alpha(\mathbf r)}{\delta n_\beta(\mathbf r')}=\frac{\delta\mu^{\rm int}_\beta(\mathbf r')}{\delta n_\alpha(\mathbf r)},\qquad\text{equivalently}\qquad c_{\alpha\beta}(\mathbf r,\mathbf r')=c_{\beta\alpha}(\mathbf r',\mathbf r).$$

In the bulk, $c_{\alpha\beta}(k)=c_{\beta\alpha}(k)$ is real and depends on $|\mathbf k|$ only.

*Proof.* For the learned part, exchange $(\alpha,\mathbf r)\leftrightarrow(\beta,\mathbf r')$ in (2.4) and relabel $i\leftrightarrow j$; the expression is unchanged because $\Phi_{ij}=\Phi_{ji}$ for a $C^2$ function (Schwarz) and because the weights are radial. For the Coulomb part the kernel $e_\alpha e_\beta/(4\pi\varepsilon_0\varepsilon|\mathbf r-\mathbf r'|)$ is manifestly symmetric. The bulk statement follows from (2.5) and (A2). $\blacksquare$

The converse explains why one should learn a scalar and not a vector.

**Lemma (Helmholtz condition).** A family of functionals $\psi_\alpha[n](\mathbf r)$ on a convex set of densities is a functional gradient, $\psi_\alpha=\delta F/\delta n_\alpha$, if and only if $\delta\psi_\alpha(\mathbf r)/\delta n_\beta(\mathbf r')=\delta\psi_\beta(\mathbf r')/\delta n_\alpha(\mathbf r)$. In that case, with $n_\lambda=n_0+\lambda(n-n_0)$,

$$F[n]=F[n_0]+\int_0^1d\lambda\sum_\alpha\int d^3r\;\psi_\alpha[n_\lambda](\mathbf r)\,\big(n_\alpha-n_{0,\alpha}\big)(\mathbf r). \tag{4.1}$$

*Proof.* Necessity is the symmetry of second derivatives. For sufficiency, differentiate (4.1):

$$\frac{\delta F}{\delta n_\beta(\mathbf r')}=\int_0^1d\lambda\Big[\psi_\beta[n_\lambda](\mathbf r')+\lambda\sum_\alpha\int d^3r\;\frac{\delta\psi_\alpha(\mathbf r)}{\delta n_\beta(\mathbf r')}\Big|_{n_\lambda}(n_\alpha-n_{0,\alpha})(\mathbf r)\Big].$$

By the symmetry assumption the second term equals $\lambda\,\partial_\lambda\psi_\beta[n_\lambda](\mathbf r')$, so the integrand is $\partial_\lambda\big(\lambda\,\psi_\beta[n_\lambda](\mathbf r')\big)$ and the integral is $\psi_\beta[n](\mathbf r')$. $\blacksquare$

Formula (4.1) is the functional line integration used by neural functionals that learn $c_1=-\beta\,\delta F_{\rm ex}/\delta n$ directly. Such a network has no reason to satisfy the Helmholtz condition exactly, and the value of (4.1) then depends on the integration path. With the ansatz (2.2) the condition holds identically.

### 4.2 Linearized theory: a rigorous statement

Linearize about a uniform state, $n_\alpha=\bar n_\alpha+\delta n_\alpha$, at $\mathbf E_0=0$. The advection term $\nabla\cdot(n_\alpha\mathbf u)$ is of second order because $\nabla\cdot\mathbf u=0$, so in Fourier space

$$\partial_t\,\delta n(\mathbf k)=A(k)\,\delta n(\mathbf k)+B(k)\,\boldsymbol\zeta(\mathbf k),\qquad A=-k^2\bar MG(k),\qquad BB^\dagger=2k_BT\,k^2\bar M, \tag{4.2}$$

$$G(k)=k_BT\,\bar N^{-1}+\tilde v^{\rm Coul}(k)+W(k),\qquad\bar N={\rm diag}(\bar n_\alpha),\qquad\bar M={\rm diag}(\mu_\alpha\bar n_\alpha).$$

$G(k)$ is the Hessian of $F$ in the bulk. The proof below uses only that $\bar M$ is symmetric positive definite, so it also covers a learned mobility matrix with cross terms.

**Proposition 2(b).** Suppose $G(k)$ is symmetric positive definite. Then (4.2) has the unique stationary covariance

$$C(k)=k_BT\,G(k)^{-1}=\big[\bar N^{-1}+\beta\tilde v^{\rm Coul}(k)+\beta W(k)\big]^{-1}, \tag{4.3}$$

which is independent of the mobilities, and the stationary process is reversible. Conversely, if the stationary covariance of $\partial_t\delta n=-k^2\bar MG\,\delta n+B\boldsymbol\zeta$ is the same for all diagonal positive $\bar M$, then $G$ is symmetric.

*Proof.* The stationary covariance solves the Lyapunov equation $AC+CA^{\top}=-BB^\dagger$. With $C=k_BTG^{-1}$ and $G=G^{\top}$, $\bar M=\bar M^{\top}$,

$$AC+CA^{\top}=-k^2k_BT\,\bar MGG^{-1}-k^2k_BT\,G^{-1}G\bar M=-2k_BT\,k^2\bar M.$$

The solution is unique because $A$ is similar to the symmetric negative definite matrix $-k^2\bar M^{1/2}G\bar M^{1/2}$ and hence has negative real eigenvalues. A stationary Ornstein–Uhlenbeck process is reversible if and only if $AC$ is symmetric, and here $AC=-k^2k_BT\bar M$. For the converse, put $X=GC$. The Lyapunov equation reads $\bar MX+X^{\top}\bar M=2k_BT\bar M$. With $\bar M={\rm diag}(m_1,m_2)$ the diagonal entries give $X_{11}=X_{22}=k_BT$, and the off-diagonal entry gives $m_1X_{12}+m_2X_{21}=0$ for all $m_1,m_2>0$, hence $X_{12}=X_{21}=0$. Therefore $C=k_BTG^{-1}$, and since a covariance is symmetric, so is $G$. $\blacksquare$

Equation (4.3) is the Ornstein–Zernike relation $S^{-1}=\bar N^{-1}-c$ with $c=-\beta(\tilde v^{\rm Coul}+W)$: the Gaussian theory reproduces exactly the structure factor encoded in the Hessian of $F$. The converse is a practical diagnostic. If a vector force is learned directly, compute the zero-field structure factor of the linearized model for two different mobility ratios; any difference measures the violation of detailed balance.

### 4.3 Nonlinear theory: a formal statement

Set $\mathbf E_0=0$ and $\mathbf u=0$. Define the mobility operator $(\mathcal M[n]\,g)_\alpha=-\nabla\cdot(\mu_\alpha n_\alpha\nabla g_\alpha)$. It is self-adjoint and positive semi-definite, since $\langle g,\mathcal Mg\rangle=\sum_\alpha\int\mu_\alpha n_\alpha|\nabla g_\alpha|^2\ge0$. Equation (1.5) reads $\partial_tn=-\mathcal M\,\delta F/\delta n+\xi$, where the noise $\xi_\alpha=\nabla\cdot(\sqrt{2k_BT\mu_\alpha n_\alpha}\,\boldsymbol\zeta_\alpha)$ has covariance $2k_BT\,\mathcal M_{\alpha\beta}(\mathbf r,\mathbf r')\,\delta(t-t')$ and $\mathcal M_{\alpha\beta}(\mathbf r,\mathbf r')=\delta_{\alpha\beta}\nabla_{\mathbf r}\!\cdot\!\nabla_{\mathbf r'}[\mu_\alpha n_\alpha(\mathbf r)\delta(\mathbf r-\mathbf r')]$ is the kernel of $\mathcal M$.

**Proposition 2(c) (formal).** The functional Fokker–Planck equation associated with (1.5) is

$$\partial_tP=\sum_{\alpha\beta}\iint d^3r\,d^3r'\;\frac{\delta}{\delta n_\alpha(\mathbf r)}\Big\{\mathcal M_{\alpha\beta}(\mathbf r,\mathbf r')\Big[\frac{\delta F}{\delta n_\beta(\mathbf r')}P+k_BT\frac{\delta P}{\delta n_\beta(\mathbf r')}\Big]\Big\}, \tag{4.4}$$

and $P_{\rm eq}[n]\propto e^{-\beta F[n]}$ is stationary with vanishing probability current.

*Proof.* The Itô Fokker–Planck equation has drift $-\mathcal M\,\delta F/\delta n$ and diffusion kernel $2k_BT\mathcal M$. Moving one functional derivative through $\mathcal M$ yields (4.4) plus the term $k_BT\,\delta/\delta n_\alpha(\mathbf r)\{P\int d^3r'\,\delta\mathcal M_{\alpha\alpha}(\mathbf r,\mathbf r')/\delta n_\alpha(\mathbf r')\}$. That term equals $\mu_\alpha\nabla_{\mathbf r}\!\cdot\!\int d^3r'\,\delta_\epsilon(\mathbf r-\mathbf r')\nabla_{\mathbf r'}\delta_\epsilon(\mathbf r-\mathbf r')$, which vanishes for any even regularization $\delta_\epsilon$ of the delta function because the integrand is odd. For $P_{\rm eq}$ the square bracket in (4.4) vanishes identically, so the probability current is zero, which together with the self-adjointness of $\mathcal M$ is detailed balance. $\blacksquare$

The argument uses integrability in an essential way. If the drift were $-\mathcal M\psi$ with $\psi$ not a functional gradient, the bracket could vanish only if $k_BT\,\delta\ln P/\delta n_\beta=-\psi_\beta$, which contradicts the Helmholtz lemma. The stationary state would then carry a probability current and produce entropy at zero field.

The statement is formal in the usual sense for Dean–Kawasaki equations: $n$ is a distribution, and $F_{\rm id}$ and the multiplicative noise need a regularization, for which the range of the weights provides a natural scale. Proposition 2(b) is free of these issues.

*Remark on the hydrodynamic coupling.* For $\mathbf u\ne0$ eliminate $\mathbf u$ with the Stokes operator $\mathcal O$ (convolution with the Oseen tensor), which is self-adjoint, positive semi-definite and annihilates gradients. The advective term becomes $-\mathcal M^{\rm hyd}\delta F/\delta n$ with $\mathcal M^{\rm hyd}_{\alpha\beta}=-\nabla\cdot n_\alpha\mathcal O\,n_\beta\nabla$, which is again a symmetric positive semi-definite Onsager mobility. Exact detailed balance at nonlinear order then requires the conjugate noise, which is the fluctuating stress of Landau and Lifshitz in (1.3). Equation (1.3) as written, and as used by Avni et al., omits it. This is a property of the original model and not of the closure, and it does not affect Proposition 2(b), because $\mathcal M^{\rm hyd}$ vanishes identically when linearized about a uniform state. If the full nonlinear model is simulated, add the fluctuating stress.

### 4.4 Fluctuation–dissipation relation

**Proposition 2(d).** In the setting of Proposition 2(b), perturb $F\to F-\sum_\alpha\int h_\alpha n_\alpha$. The response $R(k,t)=\delta\langle\delta n(\mathbf k,t)\rangle/\delta h(\mathbf k,0)$ and the stationary correlation $C(k,t)=\langle\delta n(\mathbf k,t)\,\delta n(\mathbf k,0)^\dagger\rangle$ satisfy

$$R(k,t)=-\beta\,\partial_tC(k,t),\qquad t>0.$$

*Proof.* The perturbation adds $k^2\bar Mh$ to the right-hand side of (4.2), so $R(k,t)=e^{At}k^2\bar M$. For $t>0$, $C(k,t)=e^{At}C$. Hence $-\beta\,\partial_tC(k,t)=-\beta\,e^{At}AC=-\beta\,e^{At}(-k^2k_BT\bar M)=R(k,t)$, where the detailed-balance identity $AC=-k^2k_BT\bar M$ from Proposition 2(b) was used. $\blacksquare$

The fluctuation–dissipation relation of the second kind, namely that the noise covariance $2k_BT\mathcal M$ matches the dissipative operator $\mathcal M$, holds by construction of (1.5). Together the two relations are what make the field-driven conductivity of the SDFT agree with the Green–Kubo conductivity at linear response.

## 5. Proposition 3: Noether sum rules and the stress tensor

**Proposition 3.** Under (A1), (A3), (A4), for every admissible $n$ (not only in equilibrium):

(a) the total internal force vanishes, $\sum_\alpha\int d^3r\;\mathbf f^{\rm int}_\alpha=0$;

(b) on $\mathbb R^3$ the total internal torque vanishes, $\sum_\alpha\int d^3r\;\mathbf r\times\mathbf f^{\rm int}_\alpha=0$;

(c) locally $\sum_\alpha\mathbf f^{\rm int}_\alpha=\nabla\cdot\boldsymbol\sigma^{\rm int}$ with a symmetric stress tensor $\boldsymbol\sigma^{\rm int}=\boldsymbol\sigma^{\rm M}+\boldsymbol\sigma^\theta$ given explicitly in (5.2) and (5.4).

*Proof of (a).* Take $g$ a translation by $\mathbf a$, so $(T_{\mathbf a}n)(\mathbf r)=n(\mathbf r-\mathbf a)$ and $\partial_{\mathbf a}T_{\mathbf a}n|_{\mathbf a=0}=-\nabla n$. Differentiating Proposition 1(i) at $\mathbf a=0$ and integrating by parts (boundary terms vanish by (A4)),

$$0=-\sum_\alpha\int d^3r\;\mu^{\rm int}_\alpha\nabla n_\alpha=\sum_\alpha\int d^3r\;n_\alpha\nabla\mu^{\rm int}_\alpha=-\sum_\alpha\int d^3r\;\mathbf f^{\rm int}_\alpha.\qquad\blacksquare$$

In a periodic box translations are exact symmetries, so (a) holds exactly there as well. For the ansatz (2.2) one can also verify (a) directly: $\sum_\alpha\int\mu^\theta_\alpha\nabla n_\alpha=\sum_i\int\Phi_i\nabla\bar n_i=\int\nabla\Phi_\theta(\bar n)=0$.

*Proof of (b).* Take $g$ a rotation by angle $\vartheta$ about a unit vector $\hat{\boldsymbol\omega}$. Then $\partial_\vartheta T_gn|_{\vartheta=0}=-(\hat{\boldsymbol\omega}\times\mathbf r)\cdot\nabla n$. Differentiating Proposition 1(i) and integrating by parts, using $\nabla\cdot(\hat{\boldsymbol\omega}\times\mathbf r)=0$,

$$0=-\sum_\alpha\int d^3r\;\mu^{\rm int}_\alpha\,(\hat{\boldsymbol\omega}\times\mathbf r)\cdot\nabla n_\alpha=\sum_\alpha\int d^3r\;n_\alpha\,(\hat{\boldsymbol\omega}\times\mathbf r)\cdot\nabla\mu^{\rm int}_\alpha=-\hat{\boldsymbol\omega}\cdot\sum_\alpha\int d^3r\;\mathbf r\times\mathbf f^{\rm int}_\alpha.$$

Since $\hat{\boldsymbol\omega}$ is arbitrary, (b) follows. $\blacksquare$

Parts (a) and (b) are the Noether identities of Hermann and Schmidt, applied to $F_{\rm int}$ instead of the exact excess free energy. They hold for each species sum only; the force on a single species does not vanish (Section 8, T3).

For (c) we need a classical lemma.

**Lemma (Noll).** Let $\boldsymbol{\mathfrak h}(\mathbf x,\mathbf y)=-\boldsymbol{\mathfrak h}(\mathbf y,\mathbf x)$ decay sufficiently fast in $|\mathbf x-\mathbf y|$. Then

$$\int d^3y\;\mathfrak h_p(\mathbf x,\mathbf y)=-\partial_qS_{pq}(\mathbf x),\qquad S_{pq}(\mathbf x)=\frac12\int d^3z\int_0^1d\lambda\;\mathfrak h_p\big(\mathbf x+(1-\lambda)\mathbf z,\;\mathbf x-\lambda\mathbf z\big)\,z_q. \tag{5.1}$$

*Proof.* For a test field $\boldsymbol\psi$, antisymmetry gives $\int\!\!\int\psi_p(\mathbf x)\mathfrak h_p(\mathbf x,\mathbf y)=\frac12\int\!\!\int[\psi_p(\mathbf x)-\psi_p(\mathbf y)]\,\mathfrak h_p(\mathbf x,\mathbf y)$. Write $\psi_p(\mathbf x)-\psi_p(\mathbf y)=\int_0^1d\lambda\,(x_q-y_q)\,\partial_q\psi_p(\mathbf y+\lambda(\mathbf x-\mathbf y))$ and change variables to $\mathbf z=\mathbf x-\mathbf y$, $\boldsymbol\xi=\mathbf y+\lambda\mathbf z$ (unit Jacobian). The result is $\int d^3\xi\;\partial_q\psi_p(\boldsymbol\xi)\,S_{pq}(\boldsymbol\xi)=-\int d^3\xi\;\psi_p\,\partial_qS_{pq}$. $\blacksquare$

*Proof of (c).* Coulomb part. With $\mathbf E=-\nabla\phi$, the Maxwell stress

$$\sigma^{\rm M}_{pq}=\varepsilon_0\varepsilon\big(E_pE_q-\tfrac12|\mathbf E|^2\delta_{pq}\big) \tag{5.2}$$

satisfies $\partial_q\sigma^{\rm M}_{pq}=\varepsilon_0\varepsilon\,[E_p\nabla\!\cdot\!\mathbf E+(\mathbf E\cdot\nabla)E_p-\tfrac12\partial_p|\mathbf E|^2]=\rho_ZE_p$, because $\nabla\times\mathbf E=0$ and $\varepsilon_0\varepsilon\nabla\cdot\mathbf E=\rho_Z$. This is $\sum_\alpha\mathbf f^{\rm Coul}_\alpha=-\rho_Z\nabla\phi$.

Learned part. Using (2.3) and $\nabla\Phi_\theta(\bar n(\mathbf r))=\sum_i\Phi_i\nabla\bar n_i$,

$$\sum_\alpha n_\alpha\nabla\mu^\theta_\alpha+\nabla\Phi_\theta=\int d^3r'\;\boldsymbol{\mathfrak h}(\mathbf r,\mathbf r'),\qquad\boldsymbol{\mathfrak h}(\mathbf r,\mathbf r')=\sum_{i\alpha}\nabla_{\mathbf r}w_{i\alpha}(|\mathbf r-\mathbf r'|)\,\big[n_\alpha(\mathbf r)\Phi_i(\mathbf r')+\Phi_i(\mathbf r)n_\alpha(\mathbf r')\big]. \tag{5.3}$$

The bracket is symmetric under $\mathbf r\leftrightarrow\mathbf r'$ and $\nabla_{\mathbf r}w(|\mathbf r-\mathbf r'|)$ is odd, so $\boldsymbol{\mathfrak h}$ is antisymmetric and the lemma applies. Therefore $\sum_\alpha\mathbf f^\theta_\alpha=-\sum_\alpha n_\alpha\nabla\mu^\theta_\alpha=\nabla\Phi_\theta-\int\boldsymbol{\mathfrak h}\,d^3r'=\nabla\cdot\boldsymbol\sigma^\theta$ with

$$\sigma^\theta_{pq}(\mathbf x)=\Phi_\theta(\bar n(\mathbf x))\,\delta_{pq}+\frac12\int d^3z\;\frac{z_pz_q}{|\mathbf z|}\sum_{i\alpha}w'_{i\alpha}(|\mathbf z|)\int_0^1d\lambda\;\big[n_\alpha(\mathbf x_+)\Phi_i(\mathbf x_-)+\Phi_i(\mathbf x_+)n_\alpha(\mathbf x_-)\big], \tag{5.4}$$

where $\mathbf x_+=\mathbf x+(1-\lambda)\mathbf z$ and $\mathbf x_-=\mathbf x-\lambda\mathbf z$. The tensor is symmetric because $\nabla w(|\mathbf z|)=w'(|\mathbf z|)\,\mathbf z/|\mathbf z|$ is parallel to $\mathbf z$; this is where radial weights (A1) are used, and it is the local counterpart of (b). $\blacksquare$

In a uniform state $\int d^3z\,z_pz_q\,w'(|\mathbf z|)/|\mathbf z|=-\delta_{pq}\tilde w(0)$, and (5.4) reduces to $\boldsymbol\sigma^\theta=-p^\theta\,\mathsf I$ with $p^\theta=\sum_i\bar n_i\Phi_i-\Phi_\theta$. This equals $-f+\sum_\alpha n_\alpha\mu^\theta_\alpha$, the thermodynamic pressure of the learned functional, so mechanical and thermodynamic routes to the pressure agree.

*Consequence for the Stokes equation.* The ideal-gas force is also a divergence, $-k_BT\nabla\sum_\alpha n_\alpha$. The body force in (1.3) is therefore $\nabla\cdot(\boldsymbol\sigma^{\rm M}+\boldsymbol\sigma^\theta)+\rho_Z\mathbf E_0$. The learned interactions cannot inject net momentum or angular momentum into the medium, and their isotropic part is absorbed into the pressure $p$. The only source of momentum is the external field, and for a neutral system $\int\rho_Z\mathbf E_0=0$. A directly learned vector force offers none of these guarantees: a small violation of (a) acts in (1.3) as a spurious uniform body force and drives a flow at zero field.

## 6. Proposition 4: long-wavelength electrostatics

All statements in this section are made within the linearized theory of Section 4.2 and require $G_0(k)\equiv k_BT\bar N^{-1}+W(k)$ to be positive definite at $k=0$, which is the thermodynamic stability of the short-range reference system.

### 6.1 Electroneutrality and the Stillinger–Lovett condition

**Proposition 4(a).** Under (A2), let $\mathbf e=(e_+,e_-)^{\top}$ and let $S_{ZZ}(k)=\mathbf e^{\top}C(k)\,\mathbf e$ be the charge–charge structure factor. For any learned $W$,

$$S_{ZZ}(k)=\varepsilon_0\varepsilon\,k_BT\,k^2+O(k^4),\qquad\mathbf a^{\top}C(k)\,\mathbf e=O(k^2)\ \text{ for every fixed }\mathbf a.$$

Equivalently $S_{ZZ}(k)/\sum_\alpha\bar n_\alpha e_\alpha^2\to k^2/\kappa_D^2$ with $\kappa_D^2=\sum_\alpha\bar n_\alpha e_\alpha^2/(\varepsilon_0\varepsilon k_BT)$.

*Proof.* $G=G_0+\mathbf e\mathbf e^{\top}/(\varepsilon_0\varepsilon k^2)$ is a rank-one update of $G_0$. By the Sherman–Morrison formula, with $s(k)=\mathbf e^{\top}G_0(k)^{-1}\mathbf e>0$,

$$G^{-1}=G_0^{-1}-\frac{G_0^{-1}\mathbf e\,\mathbf e^{\top}G_0^{-1}}{\varepsilon_0\varepsilon k^2+s},\qquad\mathbf e^{\top}G^{-1}\mathbf e=\frac{\varepsilon_0\varepsilon k^2\,s}{\varepsilon_0\varepsilon k^2+s},\qquad\mathbf a^{\top}G^{-1}\mathbf e=\frac{\varepsilon_0\varepsilon k^2\;\mathbf a^{\top}G_0^{-1}\mathbf e}{\varepsilon_0\varepsilon k^2+s}.$$

By (A2), $W(k)=W(0)+O(k^2)$, so $s(k)\to s(0)\in(0,\infty)$, and multiplying by $k_BT$ gives the claims. $\blacksquare$

The vanishing of $S_{ZZ}(0)$ and of all charge–density cross correlations at $k=0$ is local electroneutrality (perfect screening); the coefficient of $k^2$ is the Stillinger–Lovett second-moment condition. Neither involves $W$. In particular, ion pairing and clustering, which live in $W$, do not change the $k^2$ coefficient; they enter at $O(k^4)$ with a coefficient set by the pair size, so in a strongly coupled electrolyte the ratio $S_{ZZ}/(\varepsilon_0\varepsilon k_BTk^2)$ can deviate from 1 by $O((ka)^2)$ at the smallest wavevector of a simulation box, and the sum rule has to be tested by extrapolation in $k^2$, not read off at one point. The only way to break them is to let the network touch the $1/k^2$ singularity, which is why the Coulomb part is kept analytic. A model trained on a short-range "mimic" system alone does violate the condition, as Bui and Cox observe.

### 6.2 Debye–Hückel–Onsager limiting law

**Proposition 4(b).** Consider the charge-symmetric case $e_\pm=\pm e$, $\bar n_\pm=\bar n$, $W_{\alpha\beta}=s_\alpha s_\beta W_Z(k)$ with $s_\pm=\pm1$, and define $c(k)=1+\varepsilon_0\varepsilon k^2W_Z(k)/e^2$, so that the total interaction is $\tilde v_{\alpha\beta}=s_\alpha s_\beta\,e^2c(k)/(\varepsilon_0\varepsilon k^2)$. Assume $c$ is bounded and continuous at $k=0$. Then, as $\bar n\to0$,

$$\frac{\kappa}{\kappa_0}=1-\frac{r_s}{\lambda_D}-\frac13\Big(1-\frac1{\sqrt2}\Big)\frac{l_B}{\lambda_D}+o\big(\bar n^{1/2}\big),$$

which is the Debye–Hückel–Onsager law, independently of $W_Z$. Here $\lambda_D=(8\pi l_B\bar n)^{-1/2}$, $l_B=e^2/(4\pi\varepsilon_0\varepsilon k_BT)$ and $r_s=1/(6\pi\eta\bar\mu)$.

*Proof.* Equations (17)–(21) of Avni et al. are written in terms of $\tilde V_{\rm co}(k)$ but use only that $v_{\alpha\beta}=s_\alpha s_\beta V(r)$ with $V$ isotropic, never the specific form of $V$. Inserting $\tilde V=e^2c(k)/(\varepsilon_0\varepsilon k^2)$ in place of the cut-off potential gives their (23)–(24) with $\cos(ka)$ replaced by $c(k)$. With $x=k\lambda_D$ and $c=c(x/\lambda_D)$,

$$\frac{\kappa_{\rm hyd}}{\kappa_0}=-\frac{r_s}{\lambda_D}\,\frac2\pi\int_0^\infty dx\,\frac{c}{c+x^2},\qquad\frac{\kappa_{\rm el}}{\kappa_0}=-\frac{l_B}{\lambda_D}\,\frac1{3\pi}\int_0^\infty dx\,\frac{x^2c^2}{(x^2+c)(x^2+c/2)}.$$

Let $|c|\le c_{\max}$ and $x_0=\sqrt{2c_{\max}}$. For $x\ge x_0$ the integrands are bounded by $2c_{\max}/x^2$ and $4c_{\max}^2/x^2$. For $x\le x_0$ the argument $k=x/\lambda_D$ tends to zero uniformly, so by continuity $c\ge\frac12$ once $\lambda_D$ is large enough, and the integrands are bounded by constants. By dominated convergence the integrals tend to their values at $c\equiv1$, namely $\int_0^\infty dx/(1+x^2)=\pi/2$ and $\int_0^\infty x^2dx/[(x^2+1)(x^2+\frac12)]=\pi(1-1/\sqrt2)$. $\blacksquare$

If in addition $c(k)=1+O(k^2)$ with Gaussian-type decay, the relative correction to each integral is $O(\sigma/\lambda_D)$, where $\sigma$ is the range of the weights, so the learned part first enters $\kappa/\kappa_0$ at order $\bar n$, one order beyond the limiting law (Section 8, T6).

*General $W_{\alpha\beta}$ (argument, not a complete proof).* At the wavenumbers $k\sim\kappa_D$ that dominate both integrals, $\bar N^{-1}$ and $\beta\tilde v^{\rm Coul}$ are both of order $1/\bar n$, whereas $\beta W$ is of order one. The relative weight of the learned part in $G(k)$ is therefore $O(\bar n\,\beta W(0))$ and vanishes in the dilute limit. Turning this into a proof requires the conductivity formula for a general symmetric $2\times2$ interaction matrix, which still has to be derived.

## 7. What the construction does not guarantee

*Stability.* Nothing forces $G(k)$ to be positive definite for all $k$, and the propositions of Sections 4.2 and 6 assume it. Since $\tilde v^{\rm Coul}$ is positive semi-definite, a sufficient condition is $k_BT\bar N^{-1}+W(k)\succ0$ for all $k$, which is cheap to evaluate from (2.5) and can be added to the loss as a penalty on the smallest eigenvalue. The cut-off Coulomb potential of Avni et al. is an example of a violation: its $\cos(ka)$ makes $x^2+\cos(ax/\lambda_D)$ vanish at $a/\lambda_D\approx2.79$, a spurious charge-density-wave instability. For water with $a=0.3$ nm this lies near 8 M and is harmless; for $\varepsilon\approx10$–$20$ and $a\approx0.4$–$0.6$ nm it lies between about 0.3 and 1 M. A kernel obtained from a measured, hence positive, structure factor cannot have this problem.

*Exact detailed balance of the full nonlinear model.* This needs the fluctuating stress in (1.3); see the remark in Section 4.3.

*Mathematical well-posedness.* The diffusion coefficient $D_\alpha$ in (1.2) is a constant, but the noise amplitude is $\sqrt{2D_\alpha n_\alpha(\mathbf r,t)}$: it depends on the fluctuating field itself, because the number of independent Brownian displacements in a volume element is proportional to the local density. The noise is therefore multiplicative. For the microscopic density, a sum of delta functions, the square root and the product with white noise have no classical meaning, which is why the nonlinear equation is called formal. Linearization replaces $n_\alpha$ by $\bar n_\alpha$ under the square root, the noise becomes additive, $B=i\sqrt{2D_\alpha\bar n_\alpha}\,k$ as in (4.2), and the problem disappears. This is the setting of Avni et al. and of every rigorous statement above. A simulation of the nonlinear equation needs a regularization, for which the range of the weights is the natural scale (cells that contain many particles), and a scheme that respects the Itô form of (1.2).

*Meaning of the learned functional beyond Gaussian order.* At Gaussian order the stochastic model reproduces the structure factor encoded in the Hessian of $F$, by (4.3), so taking $W=-k_BT\,c^{\rm SR}$ from data is consistent. Beyond Gaussian order the noise renormalizes the correlations, so an equilibrium (fully renormalized) density functional used together with noise counts fluctuations twice. For a nonlinear model $F^\theta_{\rm ex}$ should be understood as the bare functional at the coarse-graining scale and trained so that the *stochastic* model reproduces the data.

*The dissipative sector.* The mobilities $\mu_\alpha$ are inputs here. Their dependence on concentration, cross-mobilities and memory are a separate modelling problem, with its own structure (a symmetric positive semi-definite mobility, or a quadratic Rayleighian in the velocities relative to the medium).

*Accuracy and coverage.* The propositions say that the model is physically admissible for every $\theta$, not that it is right.

## 8. Numerical checks

The scripts are in `verify_closure.py`. The three-dimensional tests use a $48^3$ periodic grid, two species, $K=3$ radial channels (two Gaussians and one shell-like weight, with species-dependent amplitudes of both signs), and a random $\Phi_\theta$ made of a tanh layer plus a quadratic form. Convolutions and gradients are spectral.

| Test | Proposition | Result |
| --- | --- | --- |
| T0 | (2.3) against a finite difference of $F$ | relative error $7\times10^{-9}$ |
| T1a | 1, lattice rotation, mirror and integer shift | $F$ unchanged; $\mu$ and the force magnitude map to within $10^{-14}$ |
| T1b | 1, arbitrary rotation by 0.7 rad plus shift of the analytic density | relative change of $F$ is $1\times10^{-14}$; $\int\mathbf f_+$ equals $R\int\mathbf f_+$ to all printed digits |
| T2 | 2(a), $\langle h_1,Hh_2\rangle$ against $\langle h_2,Hh_1\rangle$ across species | relative asymmetry $3\times10^{-10}$, the finite-difference floor |
| T3 | 3(a,b) | total force $\sim10^{-16}$ against a scale of 1.3; total torque $\sim10^{-11}$ against a scale of 7.9; the single-species forces are $\pm(-0.0025,\,0.0157,\,0.157)$, non-zero and opposite |
| T4 | 3(c), formula (5.4) in one dimension | $\max\lvert\partial_x\sigma-\sum_\alpha f_\alpha\rvert=7\times10^{-10}$ against $\max\lvert\sum_\alpha f_\alpha\rvert=0.11$ |
| T5 | 4(a), random indefinite $H$ | $S_{ZZ}/k^2=0.36999977$ at $k=10^{-3}$ against $\varepsilon_0\varepsilon k_BT=0.37$; cross correlation decays as $k^2$ |
| T6 | 4(b), $c=1+b\,(k\sigma)^2e^{-(k\sigma)^2/2}$ | both normalized integrals tend to 1 linearly in $\sigma/\lambda_D$, with slope $b\sqrt{2/\pi}$ for the hydrodynamic one, for $b=2$ and $b=-0.8$ |
| T7 | 2(b) and its converse | symmetric $\tilde v$: $C(k)$ identical for three mobility choices and equal to (4.3); non-symmetric $\tilde v$: $C(k)$ changes with the mobilities |

## 9. Remarks specific to salt-doped polymers

These are modelling remarks, not results.

*Hydrodynamics of the medium.* The first paper does not need it: in equilibrium $\mathbf u=0$ and (1.3) drops out (Section 10.3). For the later dynamical stage, start with $\mathbf u=0$, that is, $\kappa=\kappa_0+\kappa_{\rm el}$. This is reasonable because the matrix of a long-chain melt barely moves on the time scale of ion motion. Add the flow only if a systematic gap to the Green–Kubo conductivity remains.

If the flow is added, keep the Stokes form (1.3) and replace $\eta$ by one effective viscosity $\eta_{\rm eff}$. The macroscopic viscosity is the wrong number. In PEO-based electrolytes the conductivity is reported to stop depending on molecular weight above a few kg/mol, while the melt viscosity keeps growing, roughly as $M^{3.4}$ above entanglement. If the ions moved like spheres in a Newtonian liquid, the Stokes–Einstein relation would make $\kappa\propto1/\eta$ fall by the same factor. It does not, because the macroscopic viscosity measures how whole chains slide past each other, whereas a Li$^+$ ion, coordinated by ether oxygens, moves with the local segmental motion, which does not depend on chain length. Quantitatively, for a viscoelastic medium with relaxation modulus $G(t)$, the viscosity that enters $\kappa_{\rm hyd}$ is $\hat\eta(s)=\int_0^\infty G(t)e^{-st}dt$ evaluated at the relaxation rate of the ionic atmosphere, $s\ge\kappa_0/(\varepsilon_0\varepsilon)$, which is about 1 ns$^{-1}$ (checked on a Jeffreys fluid). In practice $\eta_{\rm eff}$ is the viscosity of an oligomer melt of the same chemistry, temperature and salt content, or the Green–Kubo stress integral truncated at about 1 ns. The choice matters: for $D=10^{-7}$ cm$^2$/s and $T=360$ K, $r_s=k_BT/(6\pi\eta D)$ is about 0.5 nm for $\eta_{\rm eff}=0.05$ Pa s and about $10^{-5}$ nm for $\eta_0=10^3$ Pa s.

Two refinements are not worth their cost here. A Brinkman term $-(\eta/\ell^2)\,\mathbf u$ in (1.3), with $\ell$ the hydrodynamic screening length, multiplies the Debye–Hückel electrophoretic term by $\ell/(\ell+\lambda_D)$. With $\lambda_D\approx0.1$–$0.3$ nm and $\ell$ of a few nanometres this factor is 0.9 or more, so it changes nothing and adds a parameter. Nonlinear rheology does not enter a linear-response coefficient at all. One caveat remains whatever is chosen: at 1 M, $\lambda_D$ is smaller than an ion, so no continuum form of the flow can be trusted quantitatively, which is one more reason to leave the term out first.

The dominant short-range physics is the coordination of the cation by the chain (ether oxygens in PEO). If the polymer is integrated out, $F^\theta_{\rm ex}[n_+,n_-]$ has to carry it. The alternative is to promote the segment density to a third field $n_{\rm p}$ with its own weighted densities in (2.1). All proofs are written for an arbitrary number of species and go through unchanged; only the species sums in Proposition 3 then include $n_{\rm p}$.

The dielectric constant is low ($\varepsilon\approx5$–$8$ for PEO), so $l_B\approx7$–$11$ nm and $l_B/a\gg2$: the system is deep in the Bjerrum pairing regime, and the linearization of Section 4.2 is quantitatively suspect even where it is stable. Two routes remain open. One is the nonlinear model with the scalar functional (2.2). The other is a chemical picture in which ion pairs are an additional species with mass-action kinetics whose equilibrium constant is derived from the same $F$.

The mobilities are tied to segmental relaxation and depend strongly on salt concentration through the glass transition temperature. This belongs to the dissipative sector of Section 7, and it is where a state-dependent memory kernel enters.

## 10. Learning from MD or CG-MD data: statics first (proposal)

This section is a plan, not a result. It was rewritten after discussion. The first step is restricted to statics, in the manner of classical density functional theory: the model learns how the ion densities respond to static external fields, and no dynamical quantity is learned.

### 10.1 Scope and the object that is learned

Apply static external potentials $V^{\rm ext}_\alpha(\mathbf r)$ to the ions, measure the mean densities $n_\alpha(\mathbf r)=\langle\hat n_\alpha(\mathbf r)\rangle$, and require the Euler–Lagrange equation

$$k_BT\ln\big(n_\alpha\Lambda_\alpha^3\big)+e_\alpha\phi[n]+\mu^\theta_\alpha[n]+V^{\rm ext}_\alpha=\mu_\alpha. \tag{10.1}$$

The object identified in this way is the equilibrium density functional of the ions with the polymer integrated out. It exists as a functional of $(n_+,n_-)$ alone because the external potentials couple to the ions only. By the Ornstein–Zernike relation its Hessian in the bulk is $k_BT\,S^{-1}(k)$, so it contains the Gaussian kernel $W(k)$ of Section 4.2; $W$ is not a separate object to be parametrized. The same functional can later drive a deterministic DDFT, or supply the kernel of the linearized SDFT.

A different object, the bare functional at a coarse-graining scale $\ell$, would be needed only for a nonlinear *stochastic* simulation, where the noise generates the correlations and an equilibrium functional would count them twice. It is defined by the particle model $U^\theta(\mathbf R)=F_{\rm int}[K_\ell\star\hat n_{\mathbf R}]$, whose Dean–Kawasaki equation is (1.5) with the same functional, and it is trained by particle-level force matching, to which the multiscale coarse-graining theorem applies because the map to the ion centres of mass is linear. It is deferred.

### 10.2 What to store

For each run, that is, for each external potential, store time-averaged one-body fields only: the density $n_\alpha(\mathbf r)$ and the internal force density

$$\mathbf f^{\rm int,MD}_\alpha(\mathbf r)=\Big\langle\sum_{j\in\alpha}\mathbf F^{\rm int}_j\,\delta(\mathbf r-\mathbf r_j)\Big\rangle, \tag{10.2}$$

where $\mathbf F^{\rm int}_j$ is the total MD force on the centre of mass of ion $j$ minus the external force. No pair correlation and no transport coefficient enters the training. In equilibrium the two stored fields are tied by the first Yvon–Born–Green equation,

$$k_BT\,\nabla n_\alpha=\mathbf f^{\rm int,MD}_\alpha-n_\alpha\nabla V^{\rm ext}_\alpha, \tag{10.3}$$

and both sides are measured. Their agreement, together with $\langle\mathbf j_\alpha\rangle=0$, is a test of equilibration, which matters for a system as slow as a polymer electrolyte. The force density is also the lower-variance estimator of the two.

Use planar potentials with a neutral and a charged part, $V^{\rm ext}_\alpha(z)=V_N(z)+e_\alpha\psi(z)$, each a random Fourier series with amplitudes from a fraction of $k_BT$ to a few $k_BT$; the lateral average then gives good statistics, and every $z$-bin is a training sample. For a planar density the radial weights reduce to

$$\bar w_{i\alpha}(z)=2\pi\int_{|z|}^\infty dr\;r\,w_{i\alpha}(r), \tag{10.4}$$

so the three-dimensional functional is trained on planar data and remains a three-dimensional functional. MD is canonical, so the $\mu_\alpha$ in (10.1) are unknown; the gradient form used below eliminates them. The functional is then determined up to a term linear in the particle numbers, which affects neither the structure nor the thermodynamic factor. The first YBG equation holds in the canonical ensemble as it does in the grand-canonical one; the canonical density profile differs from the grand-canonical one by a relative $O(1/N)$, negligible for hundreds of ions. Because each run then has its own effective $\mu_\alpha$, every baseline or prediction of a profile has to solve its Euler–Lagrange equation with the normalization constraints $\int n_\alpha=N_\alpha$ rather than at fixed $\mu$. Grand-canonical sampling, which published neural functionals rely on, is not available for a dense polymer electrolyte, so this is what makes the training possible there. The cost is set by salt diffusion over the wavelength $\lambda$ of the potential. The relaxation time of that density mode is $\lambda^2/(4\pi^2D_s)$, about 25 ns for $\lambda=3$ nm and $D_s\sim10^{-7}$ cm$^2$/s, and it grows as $\lambda^2$; a run has to cover many such times. Keep the wavelengths short, and measure $D_s$ in a pilot run first, because atomistic force fields often give ions that are ten times slower. (An earlier version of this paragraph omitted the factor $4\pi^2$ and overestimated the cost.) The concrete CG-MD setup (Kremer–Grest chains with single-bead ions, LAMMPS) is in the companion file `cgmd_setup.md`.

### 10.3 Loss

$$\mathcal L(\theta)=\sum_{\rm runs}\sum_\alpha\int d^3r\;\frac1{n_\alpha}\Big|\,\mathbf f^{\rm int,MD}_\alpha+n_\alpha\nabla\big(e_\alpha\phi[n]+\mu^\theta_\alpha[n]\big)\Big|^2. \tag{10.5}$$

This is force matching at the level of fields, and its target is literally the force density $\mathbf f^{\rm int}_\alpha$ of (1.4), the quantity that the whole construction set out to learn. It rests on the exact identity $\langle\hat{\mathbf f}^{\rm int}_\alpha\rangle=-n_\alpha\nabla(\delta F_{\rm int}/\delta n_\alpha)$ for the equilibrium functional. If the force density is not stored, (10.3) supplies it from $n_\alpha$ and the known $V^{\rm ext}_\alpha$. Add the stability penalty of Section 7.

*Why no flow term appears.* Nothing has been discarded. In equilibrium under a static potential, the species force balance (10.3) makes the total body force in the Stokes equation a pure gradient, $\sum_\alpha(\mathbf f^{\rm int}_\alpha-n_\alpha\nabla V^{\rm ext}_\alpha)=k_BT\,\nabla\sum_\alpha n_\alpha$. A gradient is balanced by the pressure, so (1.3) is solved by $\mathbf u=0$ with $p=k_BT\sum_\alpha n_\alpha$. The flux (1.2) then reduces to $\mu_\alpha\,(-k_BT\nabla n_\alpha+\mathbf f^{\rm int}_\alpha-n_\alpha\nabla V^{\rm ext}_\alpha)$, and the condition $\mathbf j_\alpha=0$ is (10.3) itself; the mobility multiplies the whole bracket and drops out. The loss (10.5) is therefore the complete stationary condition of (1.1)–(1.3), not a truncation of it. For the MD data the statement is even stronger: (10.3) is an identity of the equilibrium distribution and holds whatever the hydrodynamics of the real system is.

The converse also holds. Equilibrium data contain no information about $\mu_\alpha$, $\eta$ or the form of the momentum equation, so these cannot be learned at this stage, and they cannot contaminate $F$ either. They return in the dynamical stage, with $F$ frozen; Section 9 says how to treat the flow then.

### 10.4 What is fixed and what is learned

The decomposition of $F^\theta_{\rm ex}$ into weights and $\Phi_\theta$ is not unique: any invertible linear mixing of the channels, $\bar n\to A\bar n$, can be absorbed into $\Phi_\theta$. Only the radial shapes carry information. The economical choice is therefore to fix a radial basis $\{B_m(r)\}$ with a smooth cut-off, to form the species-resolved channels $B_m\star n_\alpha$, and to let the first, linear layer of the network learn the mixing, which is the same as learning $w_{i\alpha}=\sum_mA_{i\alpha,m}B_m$. Geometric weights of the fundamental-measure type should not be imposed: nothing in an ion–polymer system singles them out.

It helps to add an explicit pair term,

$$F^\theta_{\rm ex}=\frac12\sum_{\alpha\beta}\iint d^3r\,d^3r'\;n_\alpha(\mathbf r)\,u_{\alpha\beta}(|\mathbf r-\mathbf r'|)\,n_\beta(\mathbf r')+\int d^3r\;\Phi_\theta(\bar n), \tag{10.6}$$

with $u_{\alpha\beta}=u_{\beta\alpha}$ radial, short-ranged and expanded in the same basis. This is Avni's form with a learned kernel in place of the cut-off Coulomb correction. All propositions hold for it, by the arguments already used for $F_{\rm Coul}$, and the bulk kernel becomes $W=\tilde u+\tilde w^{\top}H\tilde w$. Weak external fields then pin $u$, and strong fields pin the nonlinear remainder $\Phi_\theta$.

### 10.5 Tests

Everything in this list is a prediction, since only one-body fields under external potentials are used for training.

Held-out external potentials, including larger amplitudes and other wavelengths.

The bulk structure factors $S_{\alpha\beta}(k)$, obtained from the Hessian of the trained functional by automatic differentiation and compared with an unperturbed run. By the static fluctuation–response relation, $\delta n_\alpha(\mathbf r)/\delta(-\beta V^{\rm ext}_\beta(\mathbf r'))=\langle\delta\hat n_\alpha(\mathbf r)\,\delta\hat n_\beta(\mathbf r')\rangle$, the structure factor *is* the linear response to an external field at every wavevector at once. It therefore probes the weak-field limit of the training set without having been part of it. For the Gaussian sector nothing is lost by comparing at this level, because the second moments are the sufficient statistic of a Gaussian field model.

The thermodynamic factor. For a 1:1 salt of density $n_s$, the constrained Hessian along the neutral direction gives

$$\Gamma\equiv1+\frac{d\ln\gamma_\pm}{d\ln n_s}=1+\frac{n_s}{2k_BT}\sum_{\alpha\beta}W_{\alpha\beta}(0),\qquad\frac{S_{NN}(0)}{2n_s}=\frac1\Gamma, \tag{10.7}$$

where $S_{NN}=\sum_{\alpha\beta}S_{\alpha\beta}$; the Coulomb term drops out by neutrality, and the second relation follows from (4.3) and the Sherman–Morrison formula. $\Gamma$ is measured for PEO/LiTFSI with concentration cells. The comparison needs care about ensembles, because a two-field description fixes the chemical potential of the polymer.

Exact conditions: the Stillinger–Lovett limit of Proposition 4(a), and path independence of the functional line integral (4.1), which a network that learns $c_1$ directly does not have.

Transfer to a non-planar geometry, for example the density around a pinned ion or a spherical external potential. A network trained on planar windows cannot be evaluated there at all; with radial weights the transfer costs nothing, by Proposition 1.

The baselines are Poisson–Boltzmann ($F^\theta_{\rm ex}=0$), a mean-spherical-approximation functional, and a planar network that learns $c_1$ directly, in the manner of Sammüller et al. and of Bui and Cox.

All of these statements are independent of mobilities, of viscosity and of the form of (1.3), by Proposition 2(b).

### 10.6 Deferred

Dynamics: mobilities and cross-mobilities, memory, the hydrodynamic term of Section 9, and the conductivity. They need time-resolved $\hat n_\alpha$ and $\hat{\mathbf j}_\alpha$. Instantaneous MD currents are inertial, so they have to be coarse-grained in time before they can be compared with an overdamped model. A CG model with soft bare interactions, for which the bare functional is known in closed form, remains a useful unit test of the Gaussian closure when that stage is reached.

## Changes relative to the earlier draft

The continuity equation had the wrong sign and a stray $\partial$; it is now (1.1)–(1.2). The weighted density was written as a product $w_i(|\mathbf r|)\,n_\alpha$ with no species sum; it is now the convolution (2.1). The interaction free energy is renamed $F_{\rm int}$ so that "ex" refers only to the learned part. The ideal-gas term is made explicit, which exposes the gradient-flow form (1.5). Assumptions (A1)–(A4) are stated, since each proof uses a specific one: radial weights for Propositions 1 and 3, $C^2$ smoothness for 2(a), short range for 4. Two claims of the draft are now qualified: "Boltzmann equilibrium exists" is rigorous for the linearized theory and formal for the nonlinear one, and needs fluctuating stress once $\mathbf u\ne0$; "the DHO law is preserved" is proved for the charge-symmetric case and argued for the general one. Stability is moved from the list of guaranteed properties to the list of things to check.

Second revision. Section 7 now says explicitly why the noise is multiplicative although $D_\alpha$ is constant, and that it becomes additive on linearization. The first remark of Section 9 claimed that $\kappa_{\rm hyd}$ drops out in a melt; that was too strong. Section 10 is new.

Third revision. Section 10 is rewritten for a statics-first plan: training uses only one-body fields (densities and force densities) under static external potentials, structure factors move from training targets to tests, the gauge redundancy between weights and $\Phi_\theta$ is stated, and an explicit learned pair term (10.6) is added. Dynamics is deferred.

Fourth revision. The hydrodynamic remark of Section 9 is cut down to a recommendation: no flow equation in the first paper; later, $\mathbf u=0$ first, then Stokes with one effective viscosity; neither a Brinkman term nor a viscoelastic equation is needed. The derivations behind it (Brinkman closed form, Laplace-transformed modulus) are summarized in one sentence each.

## References

Entries marked † are cited from memory and should be checked before use.

- Y. Avni, R. M. Adar, D. Andelman, H. Orland, *Conductivity of concentrated electrolytes*, Phys. Rev. Lett. 128, 098002 (2022); arXiv:2110.07008.
- D. S. Dean, *Langevin equation for the density of a system of interacting Langevin processes*, J. Phys. A 29, L613 (1996). †
- S. Hermann, M. Schmidt, *Noether's theorem in statistical mechanics*, Commun. Phys. 4, 176 (2021). †
- W. Noll, J. Rational Mech. Anal. 4, 627 (1955); J. H. Irving, J. G. Kirkwood, J. Chem. Phys. 18, 817 (1950). †
- F. H. Stillinger, R. Lovett, J. Chem. Phys. 49, 1991 (1968). †
- A. Donev, E. Vanden-Eijnden, *Dynamic density functional theory with hydrodynamic interactions and fluctuations*, J. Chem. Phys. 140, 234115 (2014). †
- J.-P. Péraud, A. Nonaka, J. B. Bell, A. Donev, A. L. Garcia, *Fluctuation-enhanced electric conductivity in electrolyte solutions*, PNAS 114, 10829 (2017). †
- A. J. Archer, M. Rauscher, *Dynamical density functional theory for interacting Brownian particles: stochastic or deterministic?*, J. Phys. A 37, 9325 (2004). †
- F. Sammüller, S. Hermann, D. de las Heras, M. Schmidt, *Neural functional theory for inhomogeneous fluids*, PNAS 120, e2312484120 (2023).
- A. T. Bui, S. J. Cox, *Learning classical density functionals for ionic fluids*, Phys. Rev. Lett. 134, 148001 (2025).
- J. Dijkman et al., *Learning neural free-energy functionals with pair-correlation matching*, Phys. Rev. Lett. 134, 056103 (2025); arXiv:2505.09543 for the DDFT application. † (title)
- W. G. Noid et al., *The multiscale coarse-graining method. I. A rigorous bridge between atomistic and coarse-grained models*, J. Chem. Phys. 128, 244114 (2008). †
- *Learning force field parameters from differentiable particle-field molecular dynamics*, J. Chem. Inf. Model. (2024), doi:10.1021/acs.jcim.4c00564 (title seen in search only; content not checked).
