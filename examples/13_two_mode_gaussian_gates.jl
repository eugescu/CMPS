# Example 13: beam splitter and two-mode squeezing on GridMPS product vacua.
#
# Beam splitter on identical vacua should remain essentially product. Two-mode
# squeezing should generate entanglement with a nonzero Schmidt entropy.
#
# Run:
#   julia --project=. examples/13_two_mode_gaussian_gates.jl

using Printf
include("../src/ContinuumQuadratureCMPS.jl")
using .ContinuumQuadratureCMPS

qgrid = collect(range(-8.0, 8.0; length=301))
ψvac = ComplexF64[π^(-1 / 4) * exp(-0.5 * q^2) for q in qgrid]

function report_gate(name, gate; χmax=12)
    mps = product_gridmps(qgrid, (ψvac, ψvac))
    out = apply_two_mode_gate!(mps, 1, gate; χmax)
    S = out.singular_values

    println()
    println(name)
    @printf("norm              = %.12f\n", gridmps_norm(mps))
    @printf("kept chi          = %d\n", length(S))
    @printf("truncation error  = %.6e\n", out.truncation_error)
    @printf("Schmidt entropy   = %.6e\n", schmidt_entropy(S))
    println("leading Schmidt values:")
    for k in 1:min(length(S), 8)
        @printf("  %2d  %.12e\n", k, S[k])
    end
end

report_gate("beam splitter θ=π/4 on vacuum x vacuum", BeamSplitterGate(π / 4))
report_gate("two-mode squeezing r=0.5 on vacuum x vacuum", TwoModeSqueezerGate(0.5))
