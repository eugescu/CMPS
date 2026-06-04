# Example 3: ideal GKP / Harper Hamiltonian.
# Table row: H = -ε[cos(2√π q) + cos(2√π p)] has ideal GKP code states as the
# formal ground space. The exact ground states are non-normalizable, so this
# script evaluates finite-width comb states and watches the energy approach -2ε.
#
# Run:
#   julia --project=. examples/03_ideal_gkp_hamiltonian.jl

using Printf
include("../src/ContinuumQuadratureCMPS.jl")
using .ContinuumQuadratureCMPS

ε = 1.0
α = 2sqrt(pi)
qmin, qmax = -16.0, 16.0
grid_spec = GridSpec(qmin, qmax, 2401)
qs = grid(grid_spec)

H = ideal_gkp_hamiltonian(;ε, α, boundary=:zero)

@printf("Ideal GKP Hamiltonian finite-comb scan\n")
@printf("Formal lower bound: -2ε = %.8f\n\n", -2ε)
@printf("%10s %10s %16s %14s %14s\n", "Δ", "envelope", "energy", "<cos αq>", "<cos αp>")

for Δ in [0.45, 0.35, 0.28, 0.22, 0.18]
    envelope = 0.004
    c = gkp_comb_cmps(;Δ, envelope, logical=0, nmax=16, qmin, qmax)
    ψ = wavefunction(c, qs)
    d = diagnostics(H, qs, ψ; α)
    @printf("%10.4f %10.4f %16.10f %14.8f %14.8f\n", Δ, envelope, d.energy, d.cos_αq, d.cos_αp)
end

println("\nNote: decreasing Δ alone improves q-comb sharpness but worsens p-comb overlap unless the envelope/spacing balance is tuned.")
