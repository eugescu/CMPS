# Example 17: two separated Gaussian packets in one quadrature coordinate.
#
# The state is built as an explicit two-component continuum superposition:
#
#   psi(q) = N [g(q - Q) + exp(i phi) g(q + Q)]
#
# The empty interval between the packets is cheap for this representation, while
# a uniform grid pays linearly in Q and a Fock basis pays roughly quadratically.
#
# Full run:
#   julia --project=. examples/17_two_gaussian_cat_scaling.jl
#
# Quick run is the same; this example is cheap.

using Printf
include("../src/ContinuumQuadratureCMPS.jl")
using .ContinuumQuadratureCMPS

function gaussian_packet(q, center, sigma)
    pref = (pi * sigma^2)^(-1 / 4)
    return pref * exp(-0.5 * ((q - center) / sigma)^2)
end

function cat_amplitude(q, Q, sigma, phi)
    overlap = exp(-(Q / sigma)^2)
    norm2 = 2 + 2 * cos(phi) * overlap
    return (gaussian_packet(q, Q, sigma) +
            exp(1im * phi) * gaussian_packet(q, -Q, sigma)) / sqrt(norm2)
end

function localized_cat_grid(Q, sigma; Lsigma=8.0, Nlocal=1201)
    L = Lsigma * sigma
    if Q <= L
        return collect(range(-Q - L, Q + L; length=2Nlocal))
    end

    left = collect(range(-Q - L, -Q + L; length=Nlocal))
    right = collect(range(Q - L, Q + L; length=Nlocal))
    return sort(unique(vcat(left, right)))
end

function fock_cutoff_proxy(nbar; tol=1e-6)
    z = tol <= 1e-6 ? 5.0 : 4.0
    return ceil(Int, max(1.0, nbar + z * sqrt(max(nbar, 1.0))))
end

function report_cat(Q; sigma=1.0, phi=0.0, dq_uniform=0.02)
    qgrid = localized_cat_grid(Q, sigma)
    ψ = ComplexF64[cat_amplitude(q, Q, sigma, phi) for q in qgrid]
    norm = real(trapz(qgrid, abs2.(ψ)))
    q2 = real(trapz(qgrid, qgrid.^2 .* abs2.(ψ))) / norm
    # For well-separated packets this is the exact leading photon proxy. The
    # diagnostic is intentionally about representation scaling, not optimization.
    p2 = 1 / (2sigma^2)
    nbar = photon_number_from_qp(q2, p2)

    L = 8.0 * sigma
    uniform_N = ceil(Int, (2Q + 2L) / dq_uniform) + 1
    Ncut = fock_cutoff_proxy(nbar)
    continuum_params = 5
    norm_error = abs(norm - 1)

    @printf("%11.3e %13.6e %13d %14d %8d %10.2e %8d\n",
            Q, nbar, uniform_N, Ncut, continuum_params, norm_error, length(qgrid))
end

@printf("two-Gaussian cat scaling example\n")
@printf("explicit localized components avoid paying for the empty interval\n")
@printf("%11s %13s %13s %14s %8s %10s %8s\n",
        "Q", "nbar proxy", "grid N need", "Fock proxy",
        "params", "norm err", "quad pts")

for Q in (5.0, 10.0, 1.0e3, 1.0e6)
    report_cat(Q)
end
