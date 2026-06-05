# ContinuumQuadratureCMPS

A small Julia/ITensors scaffold for a **continuous matrix-product scalar wavefunction** on one quadrature coordinate `q`:

```math
\psi(q)=v_L^\dagger U(q_{\min},q)B(q)U(q,q_{\max})v_R,
\qquad
U(a,b)=\mathcal P\exp\int_a^b A(s)\,ds.
```

This is the quadrature-space analogue of a cMPS-like continuous tensor ansatz. It is **not** the standard many-boson field cMPS.

## Install / run

```bash
cd CMPS
julia --project=.
]
instantiate
```

Then run examples:

```bash
julia --project=. examples/01_harmonic_gaussian.jl
julia --project=. examples/02_anharmonic_quartic.jl
julia --project=. examples/03_ideal_gkp_hamiltonian.jl
julia --project=. examples/04_regularized_gkp.jl
julia --project=. examples/05_constant_gauge_harmonic.jl
julia --project=. examples/06_constant_gauge_quartic.jl
julia --project=. examples/07_regularized_gkp.jl
julia --project=. examples/08_gkp_hamiltonian_noise.jl
julia --project=. examples/09_one_mode_gates.jl
julia --project=. examples/10_one_mode_gkp_errors.jl
julia --project=. examples/11_gridmps_product_states.jl
julia --project=. examples/12_cross_phase_gate.jl
julia --project=. examples/13_two_mode_gaussian_gates.jl
julia --project=. examples/14_one_mode_gkp_scaling_demo.jl
julia --project=. examples/15_high_squeezing_stress.jl
julia --project=. examples/16_large_displacement_continuum.jl
julia --project=. examples/17_two_gaussian_cat_scaling.jl
```

Run tests:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Files

- `src/ContinuumQuadratureCMPS.jl` — generic cMPS-like object, quadrature Hamiltonians, diagnostics, a finite-difference grid baseline, Optim-based scalar optimizer, ITensor contraction bridge.
- `src/Gates.jl` — one-mode continuous-variable gate definitions, inverses, and function-space/grid-backed application.
- `src/GridMPS.jl` — grid-backed MPS site/state types, product states, dense conversion, norms, one-site gate updates, and two-site weighted SVD splitting.
- `src/FockDiagnostics.jl` — Fock-basis projection, cumulative weight, cutoff, and photon-number proxy diagnostics.
- `docs/one_mode_demonstration.md` — narrative one-mode demonstration from Hamiltonians through GKP noise and one-mode gate errors.
- `test/runtests.jl` — grid helper tests, harmonic Gaussian checks, finite-difference oscillator spectrum, GKP finite-comb trend, ITensor bridge sanity check, and constant-gauge χ > 1 checks.
- `examples/01_harmonic_gaussian.jl` — Gaussian ground state of harmonic oscillator.
- `examples/02_anharmonic_quartic.jl` — quartic oscillator variational non-Gaussian trial.
- `examples/03_ideal_gkp_hamiltonian.jl` — finite combs for the formal ideal GKP/Harper Hamiltonian.
- `examples/04_regularized_gkp.jl` — optimized finite-energy approximate GKP comb for a regularized Hamiltonian.
- `examples/05_constant_gauge_harmonic.jl` — first χ > 1 constant-gauge matrix ansatz benchmarked on the harmonic oscillator.
- `examples/06_constant_gauge_quartic.jl` — χ sweep for the quartic oscillator with FD energy, overlap, normalization, and residual diagnostics.
- `examples/07_regularized_gkp.jl` — χ sweep for the finite-energy regularized GKP Hamiltonian with FD spectrum, low-energy sector overlaps, residual, and boundary diagnostics.
- `examples/08_gkp_hamiltonian_noise.jl` — χ=3 quartic Hamiltonian-noise sweep with noisy-sector and clean-code-sector fidelities.
- `examples/09_one_mode_gates.jl` — one-mode CV gate sanity checks on a grid vacuum state.
- `examples/10_one_mode_gkp_errors.jl` — X/Z displacement-error sweeps on the regularized GKP clean doublet.
- `examples/11_gridmps_product_states.jl` — product-state `GridMPS` construction and one-site gate update.
- `examples/12_cross_phase_gate.jl` — first two-mode `GridMPS` gate, diagonal cross phase with SVD truncation diagnostics.
- `examples/13_two_mode_gaussian_gates.jl` — beam splitter and two-mode squeezing on `GridMPS` product vacua.
- `examples/14_one_mode_gkp_scaling_demo.jl` — one-mode regularized-GKP scaling/compression sweep with Fock cutoff diagnostics.
- `examples/15_high_squeezing_stress.jl` — squeezed, cubic-phase, and comb-state stress test with photon/Fock diagnostics.
- `examples/16_large_displacement_continuum.jl` — huge displaced Gaussian showing that large quadrature values are cheap when the local integration window follows the packet.
- `examples/17_two_gaussian_cat_scaling.jl` — two separated Gaussian packets comparing localized continuum parameters against uniform-grid and Fock cutoff proxies.

## Benchmarks

The constant-gauge examples report the same core diagnostics. Current reference
values on the default grids are:

| Example | Target | χ | CMPS energy | FD E0 | Error | FD overlap | Residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `05_constant_gauge_harmonic.jl` | `p²/2 + q²/2` | 2 | `0.499950068579` | `0.499949994999` | `7.36e-8` | n/a | `3.87e-4` |
| `06_constant_gauge_quartic.jl` | `p²/2 + q²/2 + 0.1q⁴` | 1 | `0.560190853471` | `0.559035348215` | `1.16e-3` | `0.999792100507` | `8.19e-2` |
| `06_constant_gauge_quartic.jl` | `p²/2 + q²/2 + 0.1q⁴` | 2 | `0.559208448098` | `0.559035348215` | `1.73e-4` | `0.999968970565` | `3.39e-2` |
| `06_constant_gauge_quartic.jl` | `p²/2 + q²/2 + 0.1q⁴` | 3 | `0.559108121773` | `0.559035348215` | `7.28e-5` | `0.999991910121` | `2.58e-2` |
| `06_constant_gauge_quartic.jl` | `p²/2 + q²/2 + 0.1q⁴` | 4 | `0.559107240152` | `0.559035348215` | `7.19e-5` | `0.999992015470` | `2.56e-2` |

For `examples/07_regularized_gkp.jl`, the FD spectrum reveals a tight low-energy
doublet, so the example reports `Fm = sum(abs2(<phi_k|psi>), k=1:m)`:

| k | FD energy | Gap |
| ---: | ---: | ---: |
| 1 | `-1.248221678216` | `0.000000e+00` |
| 2 | `-1.246496671892` | `1.725006e-03` |
| 3 | `-0.572346965235` | `6.758747e-01` |
| 4 | `-0.560628158773` | `6.875935e-01` |
| 5 | `-0.521984456165` | `7.262372e-01` |
| 6 | `-0.508929848343` | `7.392918e-01` |
| 7 | `-0.006776013545` | `1.241446e+00` |
| 8 | `0.026367269026` | `1.274589e+00` |

| χ | CMPS energy | Error | F1 | F2 | F4 | F8 | Residual | Boundary |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `-0.622020906` | `6.262e-01` | `0.319840` | `0.542067` | `0.542067` | `0.917249` | `7.081e-01` | `1.675e-04` |
| 2 | `-1.189404965` | `5.882e-02` | `0.568769` | `0.963980` | `0.963980` | `0.984020` | `3.141e-01` | `1.512e-06` |
| 3 | `-1.244445169` | `3.777e-03` | `0.589288` | `0.998748` | `0.998749` | `0.999011` | `9.405e-02` | `8.993e-07` |
| 4 | `-1.244453224` | `3.768e-03` | `0.589289` | `0.998754` | `0.998754` | `0.999000` | `9.388e-02` | `8.547e-07` |

## Next steps

The scalar examples use `χ=1` embeddings so that the physics and Hamiltonian diagnostics are transparent. The constant-gauge family is the first controlled `χ > 1` bridge. The next serious step is a χ sweep for harmonic/quartic/GKP-reg, followed by `A(q)=A0+qA1+q²A2` or an RBF expansion for `A(q),B(q)`.
