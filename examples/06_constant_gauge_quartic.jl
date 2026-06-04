# Example 6: quartic oscillator constant-gauge χ sweep.
#
# Target:
#   H = p²/2 + q²/2 + λq⁴
#
# Unlike the harmonic oscillator, the quartic oscillator is non-Gaussian, so the
# auxiliary matrix capacity should start to matter. The finite-difference ground
# state is used as the benchmark for energy, overlap, and residual norm.
#
# Run:
#   julia --project=. examples/06_constant_gauge_quartic.jl

using Optim
using Printf
using Random
include("../src/ContinuumQuadratureCMPS.jl")
using .ContinuumQuadratureCMPS

Random.seed!(2)

qmin, qmax = -8.0, 8.0
λ = 0.10
grid_spec = GridSpec(qmin, qmax, 321)
Hdesc = anharmonic_hamiltonian(; λ)
H, qgrid = finite_difference_hamiltonian(Hdesc, grid_spec)
baseline = grid_eigenstates(Hdesc, grid_spec; nev=1)

E0_fd = baseline.energies[1]
ψ_fd = copy(baseline.wavefunctions[:, 1])
ψ_fd ./= sqrt(real(grid_inner(qgrid, ψ_fd, ψ_fd)))

iterations_for_chi(χ) = χ == 1 ? 700 : 2000

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

function optimize_quartic_chi(χ, H, qgrid, qmin, qmax; previous=nothing)
    fam = ConstantGaugeCMPSFamily(χ, qmin, qmax)
    loss(θ) = constant_gauge_energy(θ, fam, qgrid, H)

    starts = Vector{Vector{Float64}}()
    if previous !== nothing
        oldχ, oldθ = previous
        push!(starts, embed_theta(oldθ, oldχ, fam; noise=1e-3))
        push!(starts, embed_theta(oldθ, oldχ, fam; noise=0.05))
    end
    scale = χ == 1 ? 0.05 : 0.35
    for _ in 1:3
        push!(starts, random_constant_gauge_theta(fam; scale, gamma=0.56))
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

    θopt = bestθ
    ψ = normalized_amplitudes_constant_gauge(qgrid, θopt, fam)
    E = bestE
    return (; χ, result=best_result, theta=θopt, psi=ψ, energy=E,
            norm=real(grid_inner(qgrid, ψ, ψ)),
            residual=grid_residual_norm(qgrid, H, ψ, E))
end

@printf("quartic oscillator constant-gauge cMPS sweep\n")
@printf("λ                         = %.6f\n", λ)
@printf("finite-difference E0      = %.12f\n\n", E0_fd)
@printf("%3s %18s %18s %18s %14s %14s\n",
        "χ", "CMPS energy", "energy error", "FD overlap", "norm", "residual")

previous = Ref{Any}(nothing)
for χ in 1:4
    out = optimize_quartic_chi(χ, H, qgrid, qmin, qmax; previous=previous[])
    overlap = grid_overlap_abs2(qgrid, out.psi, ψ_fd)
    @printf("%3d %18.12f %18.6e %18.12f %14.10f %14.6e\n",
            χ, out.energy, out.energy - E0_fd, overlap, out.norm, out.residual)
    previous[] = (χ, out.theta)
end
