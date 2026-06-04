# Example 8: regularized GKP Hamiltonian-noise benchmark.
#
# Default target:
#   H(η) = Hreg + η q^4
#
# The diagnostic separates two questions:
#   Fη2      = did the χ=3 ansatz solve the noisy low-energy doublet?
#   Fclean2  = did the noisy solution remain in the clean GKP code doublet?
#
# Run:
#   julia --project=. examples/08_gkp_hamiltonian_noise.jl

using LinearAlgebra
using Optim
using Printf
using Random
include("../src/ContinuumQuadratureCMPS.jl")
using .ContinuumQuadratureCMPS

Random.seed!(5)

qmin, qmax = -10.0, 10.0
ε = 1.0
α = 2sqrt(pi)
κ = 0.05
χ = 3
noise_kind = :quartic_q
ηs = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

grid_spec = GridSpec(qmin, qmax, 401)
Hdesc = regularized_gkp_hamiltonian(; ε, κ, α, boundary=:zero)
Hclean, qgrid = finite_difference_hamiltonian(Hdesc, grid_spec)
clean_fd = grid_eigenstates(Hdesc, grid_spec; nev=8)
Eclean = clean_fd.energies
Ψclean = clean_fd.wavefunctions

function matrix_eigenstates(H, qgrid; nev::Int=8)
    nev <= length(qgrid) || error("nev must be <= grid size")
    F = eigen(Hermitian(Matrix(H)))
    vals = F.values[1:nev]
    vecs = ComplexF64.(F.vectors[:, 1:nev])
    for k in 1:nev
        vecs[:, k] ./= sqrt(real(grid_inner(qgrid, vecs[:, k], vecs[:, k])))
    end
    return (; energies=vals, wavefunctions=vecs)
end

function noise_operator(η)
    if noise_kind == :quartic_q
        return regularized_gkp_noise_operator(qgrid; kind=:quartic_q, α, λq4=η)
    elseif noise_kind == :cos2q
        return regularized_gkp_noise_operator(qgrid; kind=:cos2q, α, amp2q=η)
    elseif noise_kind == :tilt_q
        return regularized_gkp_noise_operator(qgrid; kind=:tilt_q, α, fq=η)
    elseif noise_kind == :trap_q
        return regularized_gkp_noise_operator(qgrid; kind=:trap, α, δq2=η)
    elseif noise_kind == :trap_p
        return regularized_gkp_noise_operator(qgrid; kind=:trap, α, δp2=η)
    elseif noise_kind == :phase_q
        return regularized_gkp_noise_operator(qgrid; kind=:phase_q, α, phaseq=η)
    elseif noise_kind == :phase_p
        return regularized_gkp_noise_operator(qgrid; kind=:phase_p, α, phasep=η)
    else
        error("unsupported noise_kind: $noise_kind")
    end
end

function copy_matrix_block!(dst, src, χdst, χsrc, offdst, offsrc)
    ndst = χdst^2
    nsrc = χsrc^2
    Mdst = reshape(view(dst, offdst:offdst+ndst-1), χdst, χdst)
    Msrc = reshape(view(src, offsrc:offsrc+nsrc-1), χsrc, χsrc)
    Mdst[1:χsrc, 1:χsrc] .= Msrc
    return nothing
end

function embed_theta(oldθ, oldχ, fam::ConstantGaugeCMPSFamily; noise=1e-3)
    χnew = fam.χ
    θ = noise .* randn(nparams(fam))
    old_idx = 1
    new_idx = 1

    for _ in 1:6
        copy_matrix_block!(θ, oldθ, χnew, oldχ, new_idx, old_idx)
        old_idx += oldχ^2
        new_idx += χnew^2
    end

    for _ in 1:4
        θ[new_idx:new_idx+oldχ-1] .= oldθ[old_idx:old_idx+oldχ-1]
        old_idx += oldχ
        new_idx += χnew
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

function optimize_noisy_gkp(H, qgrid, qmin, qmax, α; previous=nothing, iterations=1000)
    fam = ConstantGaugeCMPSFamily(χ, qmin, qmax)
    loss(θ) = constant_gauge_energy(θ, fam, qgrid, H)

    starts = Vector{Vector{Float64}}()
    if previous !== nothing
        push!(starts, copy(previous))
        push!(starts, previous .+ 1e-3 .* randn(length(previous)))
        push!(starts, previous .+ 2e-2 .* randn(length(previous)))
    else
        for offdiag in (-0.2, -0.5)
            push!(starts, fourier_seed_theta(fam, α; gamma=0.03, offdiag))
        end
        push!(starts, random_constant_gauge_theta(fam; scale=0.25, gamma=0.10))
    end

    best_result = nothing
    bestθ = nothing
    bestE = Inf
    for θ0 in starts
        result = optimize(loss, θ0, NelderMead(),
                          Optim.Options(iterations=iterations, show_trace=false))
        θ = Optim.minimizer(result)
        E = loss(θ)
        if E < bestE
            best_result = result
            bestθ = θ
            bestE = E
        end
    end

    ψ = normalized_amplitudes_constant_gauge(qgrid, bestθ, fam)
    return (; result=best_result, theta=bestθ, psi=ψ, energy=bestE,
            residual=grid_residual_norm(qgrid, H, ψ, bestE),
            diagnostics=gkp_diagnostics(qgrid, ψ; α, boundary=:zero))
end

@printf("regularized GKP Hamiltonian-noise benchmark\n")
@printf("noise kind                = %s\n", string(noise_kind))
@printf("χ                         = %d\n", χ)
@printf("ε                         = %.6f\n", ε)
@printf("α                         = %.12f\n", α)
@printf("κ                         = %.6f\n", κ)
@printf("clean FD E0               = %.12f\n", Eclean[1])
@printf("clean FD doublet gap      = %.6e\n", Eclean[2] - Eclean[1])
@printf("clean FD next-sector gap  = %.6e\n\n", Eclean[3] - Eclean[1])
@printf("%9s %14s %14s %10s %9s %9s %9s %10s %9s %9s %9s %9s %9s\n",
        "η", "Ecmps", "Efd", "err", "Fη2", "Fclean2", "Fclean4",
        "residual", "<cosq>", "<cosp>", "<q²>", "<p²>", "bdy")

previous = Ref{Any}(nothing)
for (iη, η) in enumerate(ηs)
    Hη = Hclean .+ noise_operator(η)
    fdη = matrix_eigenstates(Hη, qgrid; nev=4)
    iterations = iη == 1 ? 1800 : 900
    out = optimize_noisy_gkp(Hη, qgrid, qmin, qmax, α; previous=previous[], iterations)

    Fη2 = grid_subspace_overlap_abs2(qgrid, out.psi, fdη.wavefunctions; nstates=2)
    Fclean2 = grid_subspace_overlap_abs2(qgrid, out.psi, Ψclean; nstates=2)
    Fclean4 = grid_subspace_overlap_abs2(qgrid, out.psi, Ψclean; nstates=4)
    d = out.diagnostics

    @printf("%9.1e %14.9f %14.9f %10.3e %9.6f %9.6f %9.6f %10.3e %9.5f %9.5f %9.4f %9.4f %9.2e\n",
            η, out.energy, fdη.energies[1], out.energy - fdη.energies[1],
            Fη2, Fclean2, Fclean4, out.residual, d.cosq, d.cosp, d.q2, d.p2,
            d.boundary_weight)

    previous[] = out.theta
end
