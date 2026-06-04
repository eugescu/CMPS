# Example 1: harmonic oscillator.
# Table row: H = p^2/2 + q^2/2 has Gaussian ground state.
# Run from repo root:
#   julia --project=. examples/01_harmonic_gaussian.jl

using Printf
include("../src/ContinuumQuadratureCMPS.jl")
using .ContinuumQuadratureCMPS

ω = 1.0
g = GridSpec(-8, 8, 1601)
qs = grid(g)
H = harmonic_hamiltonian(;ω)

c = gaussian_cmps(;ω, qmin=g.qmin, qmax=g.qmax)
ψ = wavefunction(c, qs)
E = energy(H, qs, ψ)
d = diagnostics(H, qs, ψ)

@printf("harmonic oscillator exact E0 = %.12f\n", 0.5ω)
@printf("cMPS Gaussian energy       = %.12f\n", E)
@printf("q variance                 = %.12f\n", d.qvar)
@printf("p^2                        = %.12f\n", d.p2)

# ITensor bridge sanity check at q=0.
ψ0_matrix = amplitude(c, 0.0; Nprop=200)
ψ0_itensor = itensor_amplitude_at(c, 0.0; Nprop=200)
@printf("ψ(0) matrix evaluator      = %.12f%+.12fi\n", real(ψ0_matrix), imag(ψ0_matrix))
@printf("ψ(0) ITensor contraction   = %.12f%+.12fi\n", real(ψ0_itensor), imag(ψ0_itensor))
