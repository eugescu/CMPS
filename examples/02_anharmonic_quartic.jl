# Example 2: anharmonic oscillator.
# Table row: H = p^2/2 + q^2/2 + λ q^4 has a non-Gaussian but still ordinary
# bound-state ground wavefunction.
#
# We use a scalar cMPS trial family ψ(q) = exp(-a q^2 - b q^4), with positive
# a,b enforced by softplus.
#
# Run:
#   julia --project=. examples/02_anharmonic_quartic.jl

using Printf
include("../src/ContinuumQuadratureCMPS.jl")
using .ContinuumQuadratureCMPS

ω = 1.0
λ = 0.10
grid_spec = GridSpec(-7, 7, 1201)
H = anharmonic_hamiltonian(;ω, λ)

# Raw θ maps to positive a,b. The small floor avoids exactly flat quartic tails.
build(θ) = begin
    a = ContinuumQuadratureCMPS.softplus(θ[1]) + 1e-5
    b = ContinuumQuadratureCMPS.softplus(θ[2]) + 1e-7
    quartic_trial_cmps(a, b; qmin=grid_spec.qmin, qmax=grid_spec.qmax)
end

out = optimize_scalar_params(build, [log(exp(0.5)-1), log(exp(0.02)-1)], H, grid_spec;
                             iterations=600, show_trace=true)

a = ContinuumQuadratureCMPS.softplus(out.theta[1]) + 1e-5
b = ContinuumQuadratureCMPS.softplus(out.theta[2]) + 1e-7

@printf("optimized quartic trial parameters:\n")
@printf("  a = %.12e\n", a)
@printf("  b = %.12e\n", b)
@printf("energy = %.12f\n", out.energy)
@printf("q variance = %.12f\n", out.diagnostics.qvar)
@printf("p^2        = %.12f\n", out.diagnostics.p2)

Hgrid, qgrid = finite_difference_hamiltonian(H, grid_spec)
Egrid = rayleigh_grid_energy(qgrid, Hgrid, out.psi)
@printf("grid residual norm = %.12e\n", grid_residual_norm(qgrid, Hgrid, out.psi, Egrid))
