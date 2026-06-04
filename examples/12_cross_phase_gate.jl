# Example 12: first two-mode grid gate, cross phase.
#
# CrossPhaseGate is diagonal in the q basis, so it is the cleanest first
# two-mode TEBD-style test:
#
#   Ψ(q1,q2) -> exp(-i gamma q1 q2) Ψ(q1,q2)
#
# Run:
#   julia --project=. examples/12_cross_phase_gate.jl

using Printf
include("../src/ContinuumQuadratureCMPS.jl")
using .ContinuumQuadratureCMPS

qgrid = collect(range(-8.0, 8.0; length=301))
ψvac = ComplexF64[π^(-1 / 4) * exp(-0.5 * q^2) for q in qgrid]
γ = 0.35

println("cross-phase gate on vacuum x vacuum")
@printf("gamma = %.6f\n", γ)
@printf("%6s %8s %14s %14s %14s\n", "χmax", "χkeep", "truncation", "entropy", "norm")

for χmax in (1, 2, 4, 8)
    mps = product_gridmps(qgrid, (ψvac, ψvac))
    out = apply_two_mode_gate!(mps, 1, CrossPhaseGate(γ); χmax)
    S = out.singular_values
    @printf("%6d %8d %14.6e %14.6e %14.12f\n",
            χmax, length(S), out.truncation_error, schmidt_entropy(S), gridmps_norm(mps))
end
