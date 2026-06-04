# Example 4: regularized GKP Hamiltonian.
# Table row: H = -ε[cos(αq)+cos(αp)] + κ(q²+p²)/2 has finite-energy approximate
# GKP-like ground states.
#
# We optimize a two-parameter finite comb: peak width Δ and Gaussian envelope.
#
# Run:
#   julia --project=. examples/04_regularized_gkp.jl

using Printf
include("../src/ContinuumQuadratureCMPS.jl")
using .ContinuumQuadratureCMPS

ε = 1.0
κ = 0.02
α = 2sqrt(pi)
qmin, qmax = -14.0, 14.0
grid_spec = GridSpec(qmin, qmax, 2201)
H = regularized_gkp_hamiltonian(;ε, κ, α, boundary=:zero)

# θ -> Δ, envelope. Both positive.
build(θ) = begin
    Δ = ContinuumQuadratureCMPS.softplus(θ[1]) + 0.03
    envelope = ContinuumQuadratureCMPS.softplus(θ[2]) + 1e-5
    gkp_comb_cmps(;Δ, envelope, logical=0, nmax=14, qmin, qmax)
end

# Start from a moderately squeezed approximate comb.
θ0 = [log(exp(0.25)-1), log(exp(0.01)-1)]
out = optimize_scalar_params(build, θ0, H, grid_spec; iterations=500, show_trace=true)

Δ = ContinuumQuadratureCMPS.softplus(out.theta[1]) + 0.03
envelope = ContinuumQuadratureCMPS.softplus(out.theta[2]) + 1e-5

@printf("regularized GKP optimized finite-comb parameters:\n")
@printf("  Δ        = %.12e\n", Δ)
@printf("  envelope = %.12e\n", envelope)
@printf("energy     = %.12f\n", out.energy)
@printf("<cos αq>   = %.12f\n", out.diagnostics.cos_αq)
@printf("<cos αp>   = %.12f\n", out.diagnostics.cos_αp)
@printf("<q²>-<q>²  = %.12f\n", out.diagnostics.qvar)
@printf("<p²>       = %.12f\n", out.diagnostics.p2)

Hgrid, qgrid = finite_difference_hamiltonian(H, grid_spec)
Egrid = rayleigh_grid_energy(qgrid, Hgrid, out.psi)
@printf("grid residual norm = %.12e\n", grid_residual_norm(qgrid, Hgrid, out.psi, Egrid))
