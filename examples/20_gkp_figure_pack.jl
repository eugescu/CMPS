# Example 20: finite-energy GKP figure pack.
#
# This is a presentation-layer script for the one-mode flagship demo. It writes
# CSV data for:
#   1. q-space density curves
#   2. scaling/compression versus kappa
#   3. accuracy versus chi
#   4. Hamiltonian-noise response
#
# Full run:
#   julia --project=. examples/20_gkp_figure_pack.jl
#
# Quick smoke run:
#   CMPS_FAST_DEMO=1 julia --project=. examples/20_gkp_figure_pack.jl
#
# Optional SVG plots:
#   CMPS_PLOTS=1 julia --project=. examples/20_gkp_figure_pack.jl

using LinearAlgebra
using Optim
using Printf
using Random
include("../src/ContinuumQuadratureCMPS.jl")
using .ContinuumQuadratureCMPS

Random.seed!(20)

fast = get(ENV, "CMPS_FAST_DEMO", "0") == "1"
do_plots = get(ENV, "CMPS_PLOTS", "0") == "1"

qmin, qmax = -10.0, 10.0
ε = 1.0
α = 2sqrt(pi)
Ngrid = fast ? 241 : 401
Nfock = fast ? 80 : 220
κ_density = fast ? [0.10, 0.05] : [0.10, 0.05, 0.025]
κs = fast ? [0.10, 0.05] : [0.10, 0.07, 0.05, 0.035, 0.025]
χs = fast ? [1, 3] : [1, 2, 3, 4, 5]
ηs = fast ? [0.0, 1e-3] : [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

grid_spec = GridSpec(qmin, qmax, Ngrid)
qgrid = grid(grid_spec)

iterations_for_chi(χ) =
    fast ? (χ == 1 ? 180 : 320) : (χ == 1 ? 700 : (χ <= 3 ? 1300 : 900))

noise_iterations(iη) = fast ? (iη == 1 ? 360 : 220) : (iη == 1 ? 1800 : 900)

function write_csv(path, header, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(row, ","))
        end
    end
    return path
end

function matrix_eigenstates(H, qgrid; nev::Int=8)
    F = eigen(Hermitian(Matrix(H)))
    vals = F.values[1:nev]
    vecs = ComplexF64.(F.vectors[:, 1:nev])
    for k in 1:nev
        vecs[:, k] ./= sqrt(real(grid_inner(qgrid, vecs[:, k], vecs[:, k])))
    end
    return (; energies=vals, wavefunctions=vecs)
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

function optimize_gkp_chi(χ, H, qgrid; previous=nothing, iterations=iterations_for_chi(χ))
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
                          Optim.Options(iterations=iterations, show_trace=false))
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

function fock_cutoff_for(qgrid, ψ)
    coeffs = fock_coefficients_from_grid(qgrid, ψ; Nmax=Nfock)
    return required_fock_cutoff(coeffs; tol=1e-6), fock_weight(coeffs)
end

function maybe_plot_all(paths)
    do_plots || return nothing
    try
        @eval import DelimitedFiles
        @eval import Plots

        function readcols(path)
            raw = Base.invokelatest(DelimitedFiles.readdlm, path, ',', String)
            header = vec(raw[1, :])
            data = raw[2:end, :]
            return Dict(header[j] => parse.(Float64, data[:, j]) for j in eachindex(header))
        end

        density = readcols(paths.density)
        p_density = Base.invokelatest(Plots.plot;
                                      xlabel="q", ylabel="|ψ(q)|²",
                                      title="Finite-energy GKP density",
                                      linewidth=2)
        for κ in unique(density["kappa"])
            idx = findall(==(κ), density["kappa"])
            Base.invokelatest(Plots.plot!, p_density, density["q"][idx],
                              density["density"][idx]; label="κ=$(κ)", linewidth=2)
        end
        Base.invokelatest(Plots.savefig, p_density, paths.density_svg)

        scaling = readcols(paths.scaling)
        p_scaling = Base.invokelatest(Plots.plot, scaling["kappa"], scaling["Nfock_1e-6"];
                                      xflip=true, marker=:circle, linewidth=2,
                                      xlabel="κ", ylabel="cutoff / params",
                                      label="Fock cutoff 1e-6",
                                      title="GKP compression proxy")
        Base.invokelatest(Plots.plot!, p_scaling, scaling["kappa"], scaling["params"];
                          marker=:diamond, linewidth=2, label="CMPS params")
        Base.invokelatest(Plots.plot!, p_scaling, scaling["kappa"], scaling["nbar"];
                          marker=:square, linewidth=2, label="nbar")
        Base.invokelatest(Plots.savefig, p_scaling, paths.scaling_svg)

        accuracy = readcols(paths.accuracy)
        p_acc = Base.invokelatest(Plots.plot, accuracy["chi"], accuracy["F2"];
                                  marker=:circle, linewidth=2,
                                  xlabel="χ", ylabel="F2",
                                  label="F2", title="GKP accuracy vs χ")
        p_acc_r = Base.invokelatest(Plots.twinx, p_acc)
        Base.invokelatest(Plots.plot!, p_acc_r, accuracy["chi"], accuracy["residual"];
                          marker=:diamond, linewidth=2, linestyle=:dash,
                          ylabel="residual", label="residual")
        Base.invokelatest(Plots.savefig, p_acc, paths.accuracy_svg)

        noise = readcols(paths.noise)
        p_noise = Base.invokelatest(Plots.plot, noise["eta"], noise["Feta2"];
                                    marker=:circle, linewidth=2,
                                    xlabel="η", ylabel="sector fidelity",
                                    label="Fη2", title="GKP Hamiltonian-noise response")
        Base.invokelatest(Plots.plot!, p_noise, noise["eta"], noise["Fclean2"];
                          marker=:diamond, linewidth=2, label="Fclean2")
        Base.invokelatest(Plots.savefig, p_noise, paths.noise_svg)

        return true
    catch err
        @warn "CMPS_PLOTS=1 was set, but plotting failed. Inspect CSV output instead." exception=(err, catch_backtrace())
        return nothing
    end
end

mkpath("outputs")
paths = (;
    density=joinpath("outputs", "gkp_density_curves.csv"),
    scaling=joinpath("outputs", "gkp_scaling_compression.csv"),
    accuracy=joinpath("outputs", "gkp_accuracy_vs_chi.csv"),
    noise=joinpath("outputs", "gkp_noise_response.csv"),
    density_svg=joinpath("outputs", "gkp_density_curves.svg"),
    scaling_svg=joinpath("outputs", "gkp_scaling_compression.svg"),
    accuracy_svg=joinpath("outputs", "gkp_accuracy_vs_chi.svg"),
    noise_svg=joinpath("outputs", "gkp_noise_response.svg"),
)

println("finite-energy GKP figure pack")
println("grid points = $Ngrid, Fock Nmax = $Nfock, fast = $fast")

density_rows = Vector{Vector{Any}}()
for κ in κ_density
    Hdesc = regularized_gkp_hamiltonian(; ε, κ, α, boundary=:zero)
    fd = grid_eigenstates(Hdesc, grid_spec; nev=1)
    ψ = fd.wavefunctions[:, 1]
    for i in eachindex(qgrid)
        push!(density_rows, Any[κ, qgrid[i], abs2(ψ[i]), real(ψ[i]), imag(ψ[i])])
    end
end
write_csv(paths.density, ["kappa", "q", "density", "realpsi", "imagpsi"], density_rows)
println("wrote ", paths.density)

accuracy_rows = Vector{Vector{Any}}()
scaling_rows = Vector{Vector{Any}}()
previous_by_chi = Dict{Int,Any}()
for κ in κs
    Hdesc = regularized_gkp_hamiltonian(; ε, κ, α, boundary=:zero)
    H, _ = finite_difference_hamiltonian(Hdesc, grid_spec)
    fd = matrix_eigenstates(H, qgrid; nev=4)

    previous = nothing
    last_row = nothing
    for χ in χs
        if haskey(previous_by_chi, χ)
            previous = previous_by_chi[χ]
        end

        out = optimize_gkp_chi(χ, H, qgrid; previous)
        d = out.diagnostics
        F2 = grid_subspace_overlap_abs2(qgrid, out.psi, fd.wavefunctions; nstates=2)
        nbar = photon_number_from_qp(d.q2, d.p2)
        Ncut, fock_weight_kept = fock_cutoff_for(qgrid, out.psi)
        params = nparams(ConstantGaugeCMPSFamily(χ, qmin, qmax))

        row = Any[κ, χ, out.energy, fd.energies[1], out.energy - fd.energies[1],
                  F2, out.residual, d.cosq, d.cosp, d.q2, d.p2, nbar, Ncut,
                  fock_weight_kept, d.boundary_weight, params, Ngrid]
        push!(accuracy_rows, row)
        last_row = row

        previous = (χ, out.theta)
        previous_by_chi[χ] = previous
    end
    push!(scaling_rows, last_row)
end

header = ["kappa", "chi", "Ecmps", "Efd", "Eerr", "F2", "residual", "cosq",
          "cosp", "q2", "p2", "nbar", "Nfock_1e-6", "fock_weight",
          "boundary", "params", "grid_N"]
write_csv(paths.accuracy, header, accuracy_rows)
write_csv(paths.scaling, header, scaling_rows)
println("wrote ", paths.accuracy)
println("wrote ", paths.scaling)

noise_rows = Vector{Vector{Any}}()
κnoise = 0.05
χnoise = fast ? 3 : 3
Hdesc = regularized_gkp_hamiltonian(; ε, κ=κnoise, α, boundary=:zero)
Hclean, _ = finite_difference_hamiltonian(Hdesc, grid_spec)
clean_fd = matrix_eigenstates(Hclean, qgrid; nev=4)
previous = Ref{Any}(nothing)
for (iη, η) in enumerate(ηs)
    Hη = Hclean .+ regularized_gkp_noise_operator(qgrid; kind=:quartic_q, α, λq4=η)
    fdη = matrix_eigenstates(Hη, qgrid; nev=4)
    warm = previous[] === nothing ? nothing : (χnoise, previous[])
    out = optimize_gkp_chi(χnoise, Hη, qgrid; previous=warm,
                           iterations=noise_iterations(iη))
    Fη2 = grid_subspace_overlap_abs2(qgrid, out.psi, fdη.wavefunctions; nstates=2)
    Fclean2 = grid_subspace_overlap_abs2(qgrid, out.psi, clean_fd.wavefunctions; nstates=2)
    d = out.diagnostics
    push!(noise_rows, Any[η, out.energy, fdη.energies[1], out.energy - fdη.energies[1],
                          Fη2, Fclean2, out.residual, d.cosq, d.cosp, d.q2, d.p2,
                          d.boundary_weight])
    previous[] = out.theta
end

write_csv(paths.noise, ["eta", "Ecmps", "Efd", "Eerr", "Feta2", "Fclean2",
                        "residual", "cosq", "cosp", "q2", "p2", "boundary"],
          noise_rows)
println("wrote ", paths.noise)

if maybe_plot_all(paths) === true
    println("wrote ", paths.density_svg)
    println("wrote ", paths.scaling_svg)
    println("wrote ", paths.accuracy_svg)
    println("wrote ", paths.noise_svg)
end
