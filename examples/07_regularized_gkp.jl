# Example 7: regularized GKP constant-gauge χ sweep.
#
# Target:
#   H = -ε[cos(αq) + cos(αp)] + κ(q²+p²)/2
#
# This is the first finite-energy GKP-like benchmark for the matrix-valued
# constant-gauge family. The ideal GKP Hamiltonian remains a diagnostic target;
# this regularized model has a legitimate normalizable ground state.
#
# Run:
#   julia --project=. examples/07_regularized_gkp.jl

using Optim
using Printf
using Random
using LinearAlgebra
include("../src/ContinuumQuadratureCMPS.jl")
using .ContinuumQuadratureCMPS

Random.seed!(3)

qmin, qmax = -10.0, 10.0
ε = 1.0
α = 2sqrt(pi)
κ = 0.05

grid_spec = GridSpec(qmin, qmax, 401)
Hdesc = regularized_gkp_hamiltonian(; ε, κ, α, boundary=:zero)
H, qgrid = finite_difference_hamiltonian(Hdesc, grid_spec)
baseline = grid_eigenstates(Hdesc, grid_spec; nev=1)

E0_fd = baseline.energies[1]
ψ_fd = copy(baseline.wavefunctions[:, 1])
ψ_fd ./= sqrt(real(grid_inner(qgrid, ψ_fd, ψ_fd)))

iterations_for_chi(χ) = χ == 1 ? 900 : (χ == 3 ? 2500 : (χ == 4 ? 800 : 2000))

function copy_matrix_block!(dst, src, χdst, χsrc, offdst, offsrc)
    ndst = χdst^2
    nsrc = χsrc^2
    Mdst = reshape(view(dst, offdst:offdst+ndst-1), χdst, χdst)
    Msrc = reshape(view(src, offsrc:offsrc+nsrc-1), χsrc, χsrc)
    Mdst[1:χsrc, 1:χsrc] .= Msrc
    return nothing
end

function embed_theta(oldθ, oldχ, fam::ConstantGaugeCMPSFamily; noise=1e-3)
    χ = fam.χ
    θ = noise .* randn(nparams(fam))
    old_idx = 1
    new_idx = 1

    for _ in 1:6
        copy_matrix_block!(θ, oldθ, χ, oldχ, new_idx, old_idx)
        old_idx += oldχ^2
        new_idx += χ^2
    end

    for _ in 1:4
        θ[new_idx:new_idx+oldχ-1] .= oldθ[old_idx:old_idx+oldχ-1]
        old_idx += oldχ
        new_idx += χ
    end

    θ[end] = oldθ[end]
    return θ
end

function fourier_seed_theta(fam::ConstantGaugeCMPSFamily, α; gamma=0.03, offdiag=-0.5)
    χ = fam.χ
    θ = zeros(nparams(fam))
    idx = 1

    freqs = collect(0:χ-1) .* Float64(α)
    freqs .-= sum(freqs) / χ
    K = Matrix(Diagonal(freqs))
    θ[idx:idx+χ^2-1] .= vec(K)
    idx += χ^2

    # K_im, C_re, C_im are left zero.
    idx += 3χ^2

    B = Matrix{Float64}(I, χ, χ)
    for i in 1:χ, j in 1:χ
        if abs(i - j) == 1
            B[i, j] = offdiag
        end
    end
    θ[idx:idx+χ^2-1] .= vec(B)
    idx += 2χ^2

    θ[idx:idx+χ-1] .= 1
    idx += 2χ
    θ[idx:idx+χ-1] .= 1
    θ[end] = log(gamma)
    return θ
end

function optimize_gkp_chi(χ, H, qgrid, qmin, qmax, α; previous=nothing)
    fam = ConstantGaugeCMPSFamily(χ, qmin, qmax)
    loss(θ) = constant_gauge_energy(θ, fam, qgrid, H)

    starts = Vector{Vector{Float64}}()
    if previous !== nothing
        oldχ, oldθ = previous
        push!(starts, embed_theta(oldθ, oldχ, fam; noise=1e-3))
        push!(starts, embed_theta(oldθ, oldχ, fam; noise=0.05))
    end

    if χ > 1
        offdiags = χ == 4 ? (-0.2,) : (-0.2, -0.5)
        for offdiag in offdiags
            push!(starts, fourier_seed_theta(fam, α; gamma=0.03, offdiag))
        end
    end

    scale = χ == 1 ? 0.05 : 0.25
    gammas = χ == 1 ? (0.02, 0.05, 0.10) : (χ == 4 ? () : (0.10,))
    for gamma in gammas
        push!(starts, random_constant_gauge_theta(fam; scale, gamma))
    end

    best_result = nothing
    bestθ = nothing
    bestE = Inf
    for θ0 in starts
        result = optimize(loss, θ0, NelderMead(),
                          Optim.Options(iterations=iterations_for_chi(χ), show_trace=false))
        θ = Optim.minimizer(result)
        E = loss(θ)
        if E < bestE
            best_result = result
            bestθ = θ
            bestE = E
        end
    end

    ψ = normalized_amplitudes_constant_gauge(qgrid, bestθ, fam)
    return (; χ, result=best_result, theta=bestθ, psi=ψ, energy=bestE,
            norm=real(grid_inner(qgrid, ψ, ψ)),
            residual=grid_residual_norm(qgrid, H, ψ, bestE),
            diagnostics=gkp_diagnostics(qgrid, ψ; α, boundary=:zero))
end

@printf("regularized GKP constant-gauge cMPS sweep\n")
@printf("ε                         = %.6f\n", ε)
@printf("α                         = %.12f\n", α)
@printf("κ                         = %.6f\n", κ)
@printf("finite-difference E0      = %.12f\n\n", E0_fd)
@printf("%3s %15s %11s %12s %10s %10s %10s %10s %10s %10s\n",
        "χ", "CMPS energy", "error", "FD overlap", "resid", "<cosq>",
        "<cosp>", "<q²>", "<p²>", "bdry")

previous = Ref{Any}(nothing)
for χ in 1:4
    out = optimize_gkp_chi(χ, H, qgrid, qmin, qmax, α; previous=previous[])
    overlap = grid_overlap_abs2(qgrid, out.psi, ψ_fd)
    d = out.diagnostics
    @printf("%3d %15.9f %11.3e %12.9f %10.3e %10.6f %10.6f %10.6f %10.6f %10.3e\n",
            χ, out.energy, out.energy - E0_fd, overlap, out.residual,
            d.cosq, d.cosp, d.q2, d.p2, d.boundary_weight)
    previous[] = (χ, out.theta)
end
