# Example 14: one-mode GKP scaling/compression demo.
#
# This is the one-mode hero benchmark. It sweeps the regularized GKP envelope
# strength kappa and the constant-gauge bond dimension chi, then compares
# continuum-ansatz diagnostics against FD and Fock-projection proxies.
#
# Full run:
#   julia --project=. examples/14_one_mode_gkp_scaling_demo.jl
#
# Quick syntax/smoke run:
#   CMPS_FAST_DEMO=1 julia --project=. examples/14_one_mode_gkp_scaling_demo.jl

using LinearAlgebra
using Optim
using Printf
using Random
include("../src/ContinuumQuadratureCMPS.jl")
using .ContinuumQuadratureCMPS

Random.seed!(14)

fast = get(ENV, "CMPS_FAST_DEMO", "0") == "1"
qmin, qmax = -10.0, 10.0
ε = 1.0
α = 2sqrt(pi)
κs = fast ? [0.10, 0.05] : [0.10, 0.07, 0.05, 0.035, 0.025, 0.015]
χs = fast ? [1, 3] : [1, 2, 3, 4, 5]
Ngrid = fast ? 241 : 401
Nfock = fast ? 80 : 220

grid_spec = GridSpec(qmin, qmax, Ngrid)

iterations_for_chi(χ) =
    fast ? (χ == 1 ? 180 : 320) : (χ == 1 ? 700 : (χ <= 3 ? 1300 : 900))

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
    θ[idx:idx+χ^2-1] .= vec(Matrix(Diagonal(freqs)))
    idx += χ^2
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

function optimize_scaling_chi(χ, H, qgrid; previous=nothing)
    fam = ConstantGaugeCMPSFamily(χ, qgrid[1], qgrid[end])
    loss(θ) = constant_gauge_energy(θ, fam, qgrid, H)
    starts = Vector{Vector{Float64}}()

    if previous !== nothing
        oldχ, oldθ = previous
        if oldχ == χ
            push!(starts, copy(oldθ))
            push!(starts, oldθ .+ 1e-3 .* randn(length(oldθ)))
        else
            push!(starts, embed_theta(oldθ, oldχ, fam; noise=1e-3))
            push!(starts, embed_theta(oldθ, oldχ, fam; noise=0.03))
        end
    end

    if χ > 1
        for offdiag in (-0.2, -0.5)
            push!(starts, fourier_seed_theta(fam, α; gamma=0.03, offdiag))
        end
    end
    push!(starts, random_constant_gauge_theta(fam; scale=(χ == 1 ? 0.05 : 0.25),
                                             gamma=(χ == 1 ? 0.05 : 0.10)))

    bestθ = nothing
    bestE = Inf
    for θ0 in starts
        result = optimize(loss, θ0, NelderMead(),
                          Optim.Options(iterations=iterations_for_chi(χ),
                                        show_trace=false))
        θ = Optim.minimizer(result)
        E = loss(θ)
        if E < bestE
            bestE = E
            bestθ = θ
        end
    end

    ψ = normalized_amplitudes_constant_gauge(qgrid, bestθ, fam)
    return (; theta=bestθ, psi=ψ, energy=bestE,
            residual=grid_residual_norm(qgrid, H, ψ, bestE),
            diagnostics=gkp_diagnostics(qgrid, ψ; α, boundary=:zero))
end

function fd_from_matrix(H, qgrid; nev=4)
    F = eigen(Hermitian(Matrix(H)))
    vals = F.values[1:nev]
    vecs = ComplexF64.(F.vectors[:, 1:nev])
    for k in 1:nev
        vecs[:, k] ./= sqrt(real(grid_inner(qgrid, vecs[:, k], vecs[:, k])))
    end
    return (; energies=vals, wavefunctions=vecs)
end

@printf("one-mode regularized GKP scaling/compression demo\n")
@printf("grid points = %d, Fock Nmax = %d, fast = %s\n", Ngrid, Nfock, string(fast))
@printf("%7s %3s %13s %13s %10s %9s %10s %9s %9s %9s %9s %8s %8s %7s %6s\n",
        "κ", "χ", "Ecmps", "Efd", "Eerr", "F2", "residual", "<cosq>",
        "<cosp>", "nbar", "Nfock", "bdry", "params", "grid", "proxy")

previous_by_chi = Dict{Int,Any}()
for κ in κs
    Hdesc = regularized_gkp_hamiltonian(; ε, κ, α, boundary=:zero)
    H, qgrid = finite_difference_hamiltonian(Hdesc, grid_spec)
    fd = fd_from_matrix(H, qgrid; nev=4)

    previous = nothing
    for χ in χs
        if haskey(previous_by_chi, χ)
            previous = previous_by_chi[χ]
        end
        out = optimize_scaling_chi(χ, H, qgrid; previous)
        d = out.diagnostics
        nbar = photon_number_from_qp(d.q2, d.p2)
        coeffs = fock_coefficients_from_grid(qgrid, out.psi; Nmax=Nfock)
        Ncut = required_fock_cutoff(coeffs; tol=1e-6)
        F2 = grid_subspace_overlap_abs2(qgrid, out.psi, fd.wavefunctions; nstates=2)
        proxy = ceil(Int, 5 * max(nbar, 0.0))

        @printf("%7.3f %3d %13.7f %13.7f %10.2e %9.6f %10.2e %9.5f %9.5f %9.2f %9d %8.1e %8d %7d %6d\n",
                κ, χ, out.energy, fd.energies[1], out.energy - fd.energies[1],
                F2, out.residual, d.cosq, d.cosp, nbar, Ncut, d.boundary_weight,
                nparams(ConstantGaugeCMPSFamily(χ, qmin, qmax)), Ngrid, proxy)

        previous = (χ, out.theta)
        previous_by_chi[χ] = previous
    end
end
