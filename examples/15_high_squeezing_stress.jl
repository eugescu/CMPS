# Example 15: high-squeezing and narrow-feature one-mode stress test.
#
# This is not the hero benchmark; it is a calibration stress test for the
# quadrature backend and Fock diagnostics. It reports q-width, p-width, photon
# number proxy, and Fock cutoff estimates for Gaussian and non-Gaussian states.
#
# Full run:
#   julia --project=. examples/15_high_squeezing_stress.jl
#
# Quick smoke run:
#   CMPS_FAST_DEMO=1 julia --project=. examples/15_high_squeezing_stress.jl

using Printf
include("../src/ContinuumQuadratureCMPS.jl")
using .ContinuumQuadratureCMPS

fast = get(ENV, "CMPS_FAST_DEMO", "0") == "1"
qmin, qmax = -12.0, 12.0
Ngrid = fast ? 1201 : 4001
Nfock = fast ? 140 : 700
rs = fast ? [1.0, 2.0] : [1.0, 2.0, 3.0, 4.0]
qgrid = collect(range(qmin, qmax; length=Ngrid))

function squeezed_vacuum(qgrid, r)
    ψ = ComplexF64[π^(-1 / 4) * exp(r / 2) * exp(-0.5 * exp(2r) * q^2)
                  for q in qgrid]
    normalize_grid_state!(qgrid, ψ)
    return ψ
end

function cubic_phase_state(qgrid, r; γ=0.03)
    ψ = squeezed_vacuum(qgrid, r)
    ψ .= ComplexF64[exp(im * γ * q^3) * ψ[i] for (i, q) in enumerate(qgrid)]
    normalize_grid_state!(qgrid, ψ)
    return ψ
end

function comb_stress_state(qgrid, r)
    Δ = 0.45 * exp(-0.35r)
    c = gkp_comb_cmps(; Δ, envelope=0.004, logical=0, nmax=14,
                      qmin=qgrid[1], qmax=qgrid[end])
    ψ = wavefunction(c, qgrid)
    ψ = ComplexF64.(ψ)
    normalize_grid_state!(qgrid, ψ)
    return ψ
end

function report_state(label, scale, ψ)
    d = gkp_diagnostics(qgrid, ψ)
    nbar = photon_number_from_qp(d.q2, d.p2)
    coeffs = fock_coefficients_from_grid(qgrid, ψ; Nmax=Nfock)
    Ncut = required_fock_cutoff(coeffs; tol=1e-6)
    weight = fock_weight(coeffs)
    @printf("%18s %6.2f %10.6f %10.4e %10.4e %10.3f %8d %10.6f %7d\n",
            label, scale, d.norm, d.q2, d.p2, nbar, Ncut, weight, Ngrid)
end

@printf("high-squeezing one-mode stress test\n")
@printf("grid points = %d, Fock Nmax = %d, fast = %s\n", Ngrid, Nfock, string(fast))
@printf("%18s %6s %10s %10s %10s %10s %8s %10s %7s\n",
        "state", "scale", "norm", "<q²>", "<p²>", "nbar", "Nfock", "Fock wt", "grid")

for r in rs
    report_state("squeezed", r, squeezed_vacuum(qgrid, r))
    report_state("cubic-squeezed", r, cubic_phase_state(qgrid, r))
    report_state("GKP-comb", r, comb_stress_state(qgrid, r))
end
