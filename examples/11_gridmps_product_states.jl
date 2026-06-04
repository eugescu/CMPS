# Example 11: grid-backed MPS product states and one-site gates.
#
# This is the first many-qumode object. Each site stores a sampled
# matrix-valued function A[chiL, q, chiR]. For product states all bond
# dimensions are one.
#
# Run:
#   julia --project=. examples/11_gridmps_product_states.jl

using Printf
include("../src/ContinuumQuadratureCMPS.jl")
using .ContinuumQuadratureCMPS

qgrid = collect(range(-8.0, 8.0; length=401))
ψvac = ComplexF64[π^(-1 / 4) * exp(-0.5 * q^2) for q in qgrid]

mps = product_gridmps(qgrid, (ψvac, ψvac))
Ψ0 = gridmps_to_dense(mps)

println("vacuum x vacuum GridMPS")
@printf("norm       = %.12f\n", gridmps_norm(mps))
@printf("<q1>       = %.12f\n", dense_q_mean(qgrid, Ψ0, 1))
@printf("<q2>       = %.12f\n", dense_q_mean(qgrid, Ψ0, 2))

apply_one_mode_gate!(mps, 1, XDisplacementGate(1.0))
Ψx = gridmps_to_dense(mps)

println()
println("after X(1.0) on site 1")
@printf("norm       = %.12f\n", gridmps_norm(mps))
@printf("<q1>       = %.12f\n", dense_q_mean(qgrid, Ψx, 1))
@printf("<q2>       = %.12f\n", dense_q_mean(qgrid, Ψx, 2))
