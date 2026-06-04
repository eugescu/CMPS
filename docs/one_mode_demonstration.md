# One-Mode Demonstration

## Goal

This repository builds a small continuum tensor-network playground for one
qumode. The working object is a continuum wavefunction in one quadrature
coordinate, evaluated through grid quadrature when we need concrete numerical
energies, overlaps, gates, and diagnostics.

The grid is not the ansatz; it is the quadrature backend used to evaluate
continuum states, Hamiltonians, gates, and diagnostics.

## Harmonic Oscillator Sanity Check

The first check is the harmonic oscillator,

```text
H = p^2/2 + q^2/2.
```

The Gaussian scalar cMPS matches the expected ground-state energy
`E0 = 0.5`, and the finite-difference grid baseline gives the same reference.
This validates the normalization, quadrature, and derivative conventions before
using more structured ansaetze.

Relevant files:

- `examples/01_harmonic_gaussian.jl`
- `examples/05_constant_gauge_harmonic.jl`

## Quartic Oscillator

The quartic oscillator,

```text
H = p^2/2 + q^2/2 + 0.1 q^4,
```

is the first non-Gaussian variational target. The constant-gauge matrix family
shows the expected trend: increasing bond dimension improves the energy,
overlap with the finite-difference ground state, and residual norm.

Representative default-grid values:

| chi | CMPS energy | FD E0 | error | FD overlap | residual |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.560190853471` | `0.559035348215` | `1.16e-3` | `0.999792100507` | `8.19e-2` |
| 2 | `0.559208448098` | `0.559035348215` | `1.73e-4` | `0.999968970565` | `3.39e-2` |
| 3 | `0.559108121773` | `0.559035348215` | `7.28e-5` | `0.999991910121` | `2.58e-2` |
| 4 | `0.559107240152` | `0.559035348215` | `7.19e-5` | `0.999992015470` | `2.56e-2` |

Relevant file:

- `examples/06_constant_gauge_quartic.jl`

## Regularized GKP Hamiltonian

The regularized GKP benchmark uses

```text
H = -cos(alpha q) - cos(alpha p) + 0.05 (q^2 + p^2)/2.
```

The important diagnostic correction is that the lowest two finite-difference
states form a tight doublet:

```text
E2 - E1 = 1.725006e-03
E3 - E1 = 6.758747e-01
```

That means a single ground-vector overlap `F1` is too brittle. A variational
state can be physically good if it lands in the protected low-energy sector
without matching the arbitrary FD eigenvector orientation inside the doublet.
For this benchmark, the meaningful code-sector metric is

```text
F2 = sum_{k=1}^2 |<phi_k | psi>|^2.
```

The default chi sweep makes the point:

| chi | energy error | F1 | F2 | residual |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `6.262e-01` | `0.319840` | `0.542067` | `7.081e-01` |
| 2 | `5.882e-02` | `0.568769` | `0.963980` | `3.141e-01` |
| 3 | `3.777e-03` | `0.589288` | `0.998748` | `9.405e-02` |
| 4 | `3.768e-03` | `0.589289` | `0.998754` | `9.388e-02` |

The chi=3 ansatz is not failing to find the regularized GKP state; it is finding
the low-energy doublet rather than the particular FD vector labeled `phi_1`.

Relevant file:

- `examples/07_regularized_gkp.jl`

## Hamiltonian Noise

The Hamiltonian-noise benchmark distinguishes two fidelities:

```text
Feta2     = overlap with the noisy Hamiltonian's low-energy doublet
Fclean2   = overlap with the original clean GKP doublet
```

This separates solve quality from code-sector damage. In the quartic distortion
benchmark,

```text
H(eta) = Hreg + eta q^4,
```

`Feta2` stays high when the ansatz follows the noisy low-energy sector, while
`Fclean2` decays as the noisy state leaves the clean code sector.

Relevant file:

- `examples/08_gkp_hamiltonian_noise.jl`

## One-Mode Gates

The one-mode gate layer adds circuit-style operations on grid-backed
wavefunctions:

| gate | action |
| --- | --- |
| `XDisplacementGate(s)` | `psi(q) -> psi(q - s)` |
| `ZDisplacementGate(t)` | `psi(q) -> exp(i t q) psi(q)` |
| `WeylDisplacementGate(s,t)` | `psi(q) -> exp(i t (q - s/2)) psi(q - s)` |
| `QuadraticPhaseGate(gamma)` | `psi(q) -> exp(i gamma q^2/2) psi(q)` |
| `CubicPhaseGate(gamma)` | `psi(q) -> exp(i gamma q^3) psi(q)` |
| `SqueezeGate(r)` | `psi(q) -> exp(r/2) psi(exp(r) q)` |

The tests verify the Weyl relation

```text
Z(t) X(s) = exp(i s t) X(s) Z(t),
```

and verify that each one-mode unitary approximately undoes itself through
`inverse_gate`. Phase gates roundtrip at very tight tolerance; X and squeezing
are limited by interpolation.

The GKP displacement-error example then applies X and Z shifts to the clean FD
regularized-GKP state. X displacements damage `<cos alpha q>` while mostly
leaving `<cos alpha p>` fixed; Z displacements do the complementary thing. This
is the first one-mode CV error-channel diagnostic.

Relevant files:

- `src/Gates.jl`
- `examples/09_one_mode_gates.jl`
- `examples/10_one_mode_gkp_errors.jl`

## Conclusion

The one-mode layer is now validated enough to become the physical-index update
inside a grid-backed MPS. The next layer is many-qumode mechanics:

```text
GridMPSSite: A[chiL, q, chiR]
one-mode gate: update one site's q index
two-mode gate: contract two sites, apply a gate, split with weighted SVD
```

That is the transition from one-qumode continuum variational examples to
TEBD-style multi-qumode circuit dynamics.
