# Example 9: one-mode continuous-variable gates on grid wavefunctions.
#
# This is the first circuit-model layer: gate objects act on a single qumode
# wavefunction ψ(q). Later, the same one-site update can be lifted to each
# virtual matrix element of a grid-backed MPS site tensor.
#
# Run:
#   julia --project=. examples/09_one_mode_gates.jl

using LinearAlgebra
using Printf
include("../src/ContinuumQuadratureCMPS.jl")
using .ContinuumQuadratureCMPS

qmin = -10.0
qmax = 10.0
N = 1001

qgrid = collect(range(qmin, qmax; length=N))
ψvac = ComplexF64[π^(-1 / 4) * exp(-0.5 * q^2) for q in qgrid]
normalize_grid_state!(qgrid, ψvac)

function grid_state_inner(qgrid, ψ, ϕ)
    dq = qgrid[2] - qgrid[1]
    return sum(conj.(ψ) .* ϕ) * dq
end

function report_state(name, qgrid, ψ)
    dq = qgrid[2] - qgrid[1]
    norm2 = real(sum(abs2, ψ) * dq)
    qmean = real(sum(conj.(ψ) .* (qgrid .* ψ)) * dq)
    q2 = real(sum(abs2.(qgrid) .* abs2.(ψ)) * dq)

    println()
    println(name)
    @printf("norm     = %.12f\n", norm2)
    @printf("<q>      = %.12f\n", qmean)
    @printf("<q²>     = %.12f\n", q2)
end

report_state("vacuum", qgrid, ψvac)

gates = [
    XDisplacementGate(1.0),
    ZDisplacementGate(0.7),
    WeylDisplacementGate(1.0, 0.7),
    QuadraticPhaseGate(0.25),
    CubicPhaseGate(0.05),
    SqueezeGate(0.4),
]

for gate in gates
    ψg = apply_gate_to_grid(gate, qgrid, ψvac)
    report_state(string(typeof(gate)), qgrid, ψg)
    F = abs2(grid_state_inner(qgrid, ψvac, ψg))
    @printf("fidelity with vacuum = %.12f\n", F)
end
