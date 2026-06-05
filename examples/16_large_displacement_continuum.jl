# Example 16: huge displacement in an unbounded quadrature coordinate.
#
# This example is intentionally analytic. A displaced oscillator ground state
# remains trivial in a continuum representation even when its center Q would
# require an enormous Fock cutoff.
#
# Full run:
#   julia --project=. examples/16_large_displacement_continuum.jl
#
# Quick run is the same; this example is cheap.

using Printf
include("../src/ContinuumQuadratureCMPS.jl")
using .ContinuumQuadratureCMPS

function displaced_gaussian(xgrid)
    ψ = ComplexF64[pi^(-1 / 4) * exp(-0.5 * x^2) for x in xgrid]
    norm = sqrt(real(trapz(xgrid, abs2.(ψ))))
    ψ ./= norm
    return ψ
end

function fock_cutoff_proxy(nbar; tol=1e-6)
    z = tol <= 1e-6 ? 5.0 : 4.0
    return ceil(Int, max(1.0, nbar + z * sqrt(max(nbar, 1.0))))
end

function report_displacement(Q; L=8.0, Ngrid=1601)
    xgrid = collect(range(-L, L; length=Ngrid))
    qgrid = Q .+ xgrid
    ψ = displaced_gaussian(xgrid)
    density = abs2.(ψ)
    qmean = real(trapz(xgrid, qgrid .* density))
    q2 = real(trapz(xgrid, qgrid.^2 .* density))
    p2 = 0.5
    nbar = max(0.0, photon_number_from_qp(q2, p2))
    Ncut = fock_cutoff_proxy(nbar)
    continuum_params = 1

    @printf("%11.3e %13.6e %13.6e %10.6f %13.6e %14d %8d [% .6e,% .6e]\n",
            Q, qmean, q2, p2, nbar, Ncut, continuum_params,
            qgrid[1], qgrid[end])
end

@printf("large-displacement continuum example\n")
@printf("local grid follows the packet, so resolution is independent of |Q|\n")
@printf("%11s %13s %13s %10s %13s %14s %8s %31s\n",
        "Q", "<q>", "<q^2>", "<p^2>", "nbar", "Fock proxy",
        "params", "grid window")

for Q in (0.0, 1.0e3, 1.0e6)
    report_displacement(Q)
end
