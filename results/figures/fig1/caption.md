**Fig. 1 | A learned density functional for the ions of a salt-doped polymer melt.**

**A, Simulation data.** Molecular dynamics under a planar external potential provides ion densities n_±(z) and internal force densities f_±^int(z). Error bars are estimated from eight trajectory blocks.

**B, Functional and training.** Learned radial convolutions and a pointwise neural network parameterize the excess free energy, which is trained by matching internal force densities to MD. The ideal-gas and Coulomb contributions are included analytically.

**C, Predictions.** Predicted density profiles for a held-out run (c = 0.04), bulk charge structure, and number response Γ(k₁) = 2n̄/S_NN(k₁) at the smallest nonzero wave number k₁, compared with MD. The bulk quantities follow from the second functional derivative and are not training targets.
