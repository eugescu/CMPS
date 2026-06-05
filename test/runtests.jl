using ContinuumQuadratureCMPS
using LinearAlgebra
using Random
using Test

@testset "grid helpers" begin
    @test_throws ErrorException GridSpec(-1, 1, 4)
    @test_throws ErrorException GridSpec(1, -1, 10)

    g = GridSpec(-1, 1, 11)
    qs = grid(g)
    @test length(qs) == 11
    @test trapz(qs, ones(length(qs))) ≈ 2.0
end

@testset "harmonic Gaussian" begin
    g = GridSpec(-8, 8, 801)
    qs = grid(g)
    H = harmonic_hamiltonian(; ω=1.0)
    c = gaussian_cmps(; ω=1.0, qmin=g.qmin, qmax=g.qmax)
    ψ = wavefunction(c, qs)

    @test energy(H, qs, ψ) ≈ 0.5 atol=1e-4
    @test diagnostics(H, qs, ψ).qvar ≈ 0.5 atol=1e-8

    ψ0_matrix = amplitude(c, 0.0; Nprop=201)
    ψ0_itensor = itensor_amplitude_at(c, 0.0; Nprop=201)
    @test ψ0_itensor ≈ ψ0_matrix atol=1e-12
end

@testset "finite-difference oscillator baseline" begin
    g = GridSpec(-8, 8, 321)
    H = harmonic_hamiltonian(; ω=1.0)
    out = grid_eigenstates(H, g; nev=3)

    @test out.energies[1] ≈ 0.5 atol=2e-3
    @test out.energies[2] ≈ 1.5 atol=4e-3
    @test out.energies[3] ≈ 2.5 atol=8e-3
    @test trapz(out.q, abs2.(out.wavefunctions[:, 1])) ≈ 1.0 atol=1e-10
    @test grid_subspace_overlap_abs2(out.q, out.wavefunctions[:, 1], out.wavefunctions;
                                     nstates=3) ≈ 1.0 atol=1e-10
end

@testset "GKP finite-comb trend" begin
    α = 2sqrt(pi)
    g = GridSpec(-14, 14, 1201)
    qs = grid(g)
    H = ideal_gkp_hamiltonian(; ε=1.0, α, boundary=:zero)

    wide = wavefunction(gkp_comb_cmps(; Δ=0.45, envelope=0.004, nmax=12,
                                      qmin=g.qmin, qmax=g.qmax), qs)
    narrow = wavefunction(gkp_comb_cmps(; Δ=0.22, envelope=0.004, nmax=12,
                                        qmin=g.qmin, qmax=g.qmax), qs)

    @test diagnostics(H, qs, narrow; α).energy < diagnostics(H, qs, wide; α).energy
end

@testset "constant gauge family" begin
    rng = MersenneTwister(11)
    fam = ConstantGaugeCMPSFamily(2, -4.0, 4.0)
    θ = random_constant_gauge_theta(fam; scale=0.01, gamma=0.5, rng)

    @test length(θ) == nparams(fam)

    A, B, vL, vR, γ = unpack_constant_gauge(θ, fam)
    @test size(A) == (2, 2)
    @test size(B) == (2, 2)
    @test length(vL) == 2
    @test length(vR) == 2
    @test γ > 0
    @test tr(A) ≈ 0 atol=1e-12

    qgrid = collect(range(-4.0, 4.0; length=101))
    ψ = amplitudes_constant_gauge(qgrid, θ, fam)

    @test length(ψ) == length(qgrid)
    @test all(isfinite, real.(ψ))
    @test all(isfinite, imag.(ψ))

    ψn = normalized_amplitudes_constant_gauge(qgrid, θ, fam)
    dq = qgrid[2] - qgrid[1]
    @test real(sum(abs2, ψn) * dq) ≈ 1.0 atol=1e-8
end

@testset "constant gauge harmonic energy finite" begin
    rng = MersenneTwister(13)
    g = GridSpec(-6.0, 6.0, 151)
    H, qgrid = finite_difference_hamiltonian(harmonic_hamiltonian(), g)

    fam = ConstantGaugeCMPSFamily(2, g.qmin, g.qmax)
    θ = random_constant_gauge_theta(fam; scale=0.01, gamma=0.5, rng)

    E = constant_gauge_energy(θ, fam, qgrid, H)

    @test isfinite(E)
    @test E > 0
end

@testset "grid overlap and residual diagnostics" begin
    g = GridSpec(-6.0, 6.0, 151)
    Hdesc = harmonic_hamiltonian()
    H, qgrid = finite_difference_hamiltonian(Hdesc, g)
    out = grid_eigenstates(Hdesc, g; nev=1)
    ψ = copy(out.wavefunctions[:, 1])
    ψ ./= sqrt(real(grid_inner(qgrid, ψ, ψ)))

    @test real(grid_inner(qgrid, ψ, ψ)) ≈ 1.0 atol=1e-10
    @test grid_overlap_abs2(qgrid, ψ, ψ) ≈ 1.0 atol=1e-10
    @test grid_residual_norm(qgrid, H, ψ, out.energies[1]) < 1e-10
end

@testset "quartic variational sanity" begin
    λ = 0.10
    g = GridSpec(-7.0, 7.0, 181)
    qgrid = grid(g)
    Hdesc = anharmonic_hamiltonian(; λ)
    H, _ = finite_difference_hamiltonian(Hdesc, g)
    baseline = grid_eigenstates(Hdesc, g; nev=1)
    Efd = baseline.energies[1]

    ψ_gauss = wavefunction(gaussian_cmps(; qmin=g.qmin, qmax=g.qmax), qgrid)
    ψ_gauss ./= sqrt(real(grid_inner(qgrid, ψ_gauss, ψ_gauss)))
    Egauss = rayleigh_grid_energy(qgrid, H, ψ_gauss)

    ψ_quartic = wavefunction(quartic_trial_cmps(0.5614, 0.0189; qmin=g.qmin, qmax=g.qmax), qgrid)
    ψ_quartic ./= sqrt(real(grid_inner(qgrid, ψ_quartic, ψ_quartic)))
    Equartic = rayleigh_grid_energy(qgrid, H, ψ_quartic)

    rng = MersenneTwister(17)
    fam = ConstantGaugeCMPSFamily(2, g.qmin, g.qmax)
    θ = random_constant_gauge_theta(fam; scale=0.01, gamma=0.5, rng)
    ψ = normalized_amplitudes_constant_gauge(qgrid, θ, fam)
    E = constant_gauge_energy(θ, fam, qgrid, H)

    @test Efd < Equartic < Egauss
    @test isfinite(E)
    @test real(grid_inner(qgrid, ψ, ψ)) ≈ 1.0 atol=1e-8
    @test isfinite(grid_residual_norm(qgrid, H, ψ, E))
end

@testset "regularized GKP grid diagnostics" begin
    α = 2sqrt(pi)
    g = GridSpec(-8.0, 8.0, 161)
    qgrid = grid(g)
    Hdesc = regularized_gkp_hamiltonian(; ε=1.0, κ=0.05, α, boundary=:zero)
    H, _ = finite_difference_hamiltonian(Hdesc, g)
    @test norm(H - H') / max(norm(H), eps()) < 1e-10

    out = grid_eigenstates(Hdesc, g; nev=1)
    ψ = copy(out.wavefunctions[:, 1])
    ψ ./= sqrt(real(grid_inner(qgrid, ψ, ψ)))
    @test grid_subspace_overlap_abs2(qgrid, ψ, out.wavefunctions; nstates=1) ≈ 1.0 atol=1e-10

    T = translation_matrix(qgrid, α; boundary=:zero)
    shifted = T * ψ
    @test shifted ≈ ContinuumQuadratureCMPS.interp_shift(qgrid, ψ, α; boundary=:zero)

    Cp = cos_p_matrix(qgrid, α; boundary=:zero)
    @test size(Cp) == size(H)
    @test norm(Cp - Cp') / max(norm(Cp), eps()) < 1e-10

    Cphase = cos_p_phase_matrix(qgrid, α, 0.1; boundary=:zero)
    @test size(Cphase) == size(H)
    @test norm(Cphase - Cphase') / max(norm(Cphase), eps()) < 1e-10

    noise_ops = (
        regularized_gkp_noise_operator(qgrid; kind=:tilt_q, α, fq=1e-3),
        regularized_gkp_noise_operator(qgrid; kind=:trap, α, δq2=1e-3, δp2=2e-3),
        regularized_gkp_noise_operator(qgrid; kind=:quartic_q, α, λq4=1e-4),
        regularized_gkp_noise_operator(qgrid; kind=:cos2q, α, amp2q=1e-3),
        regularized_gkp_noise_operator(qgrid; kind=:phase_q, α, phaseq=0.05),
        regularized_gkp_noise_operator(qgrid; kind=:phase_p, α, phasep=0.05),
    )
    for Nop in noise_ops
        @test size(Nop) == size(H)
        @test norm(Nop - Nop') / max(norm(Nop), eps()) < 1e-10
    end

    d = gkp_diagnostics(qgrid, ψ; α, boundary=:zero)
    @test d.norm ≈ 1.0 atol=1e-10
    @test isfinite(d.cosq)
    @test isfinite(d.cosp)
    @test d.q2 > 0
    @test d.p2 > 0
    @test 0 <= d.boundary_weight <= 1
end

@testset "large displacement continuum example" begin
    Q = 1.0e6
    L = 8.0
    N = 2001

    xgrid = collect(range(-L, L; length=N))
    qgrid = Q .+ xgrid
    dx = xgrid[2] - xgrid[1]

    ψ = ComplexF64[π^(-1 / 4) * exp(-0.5 * x^2) for x in xgrid]
    ψ ./= sqrt(real(sum(abs2, ψ) * dx))

    norm_check = real(sum(abs2, ψ) * dx)
    qmean = real(sum(qgrid .* abs2.(ψ)) * dx)
    centered_q2 = real(sum(abs2.(xgrid) .* abs2.(ψ)) * dx)

    Hfree, _ = finite_difference_hamiltonian(
        Hamiltonian1D("free kinetic", 0.5, q -> 0.0, Tuple{ComplexF64,Float64}[], :zero),
        GridSpec(-L, L, N),
    )
    p2 = 2 * grid_expectation(xgrid, Hfree, ψ)

    @test isapprox(norm_check, 1.0; atol=1e-10)
    @test isapprox(qmean, Q; atol=1e-4)
    @test isapprox(centered_q2, 0.5; atol=1e-5)
    @test isapprox(p2, 0.5; atol=5e-3)

    nbar_proxy = 0.5 * Q^2
    @test isapprox(nbar_proxy, 5.0e11; rtol=1e-14)
end

@testset "two-Gaussian cat scaling example" begin
    function gaussian_packet_grid(qgrid; center::Real=0.0, σ::Real=1.0)
        return ComplexF64[
            (π * σ^2)^(-1 / 4) * exp(-0.5 * ((q - center) / σ)^2)
            for q in qgrid
        ]
    end

    uniform_grid_points_for_cat(Q; halfwidth=8.0, dq=0.02) =
        ceil(Int, (2Q + 2halfwidth) / dq) + 1

    Q = 3.0
    σ = 1.0
    L = 8.0
    N = 4001
    qgrid = collect(range(-Q - L, Q + L; length=N))
    dq = qgrid[2] - qgrid[1]

    ψp = gaussian_packet_grid(qgrid; center=Q, σ)
    ψm = gaussian_packet_grid(qgrid; center=-Q, σ)

    normalize_grid_state!(qgrid, ψp)
    normalize_grid_state!(qgrid, ψm)

    S_grid = real(grid_inner(qgrid, ψm, ψp))
    S_exact = exp(-Q^2 / σ^2)

    @test isapprox(S_grid, S_exact; rtol=1e-3, atol=1e-8)

    ψcat = (ψp .+ ψm) ./ sqrt(2 + 2S_exact)
    normalize_grid_state!(qgrid, ψcat)

    norm_check = real(sum(abs2, ψcat) * dq)
    qmean = real(sum(qgrid .* abs2.(ψcat)) * dq)

    @test isapprox(norm_check, 1.0; atol=1e-10)
    @test abs(qmean) < 1e-8

    Qs = [1.0, 10.0, 100.0, 1.0e3, 1.0e6]
    gridNs = [uniform_grid_points_for_cat(Q; halfwidth=8.0, dq=0.02) for Q in Qs]
    fock_proxies = [0.5 * Q^2 for Q in Qs]

    @test all(diff(gridNs) .> 0)
    @test all(diff(fock_proxies) .> 0)

    localized_param_count = fill(6, length(Qs))
    @test all(localized_param_count .== localized_param_count[1])
end

@testset "spatial bipartition entropy identities" begin
    erf_float_test(x) = ccall(:erf, Float64, (Float64,), Float64(x))
    binary_entropy_test(p) = begin
        p = clamp(Float64(p), 0.0, 1.0)
        (p == 0.0 || p == 1.0) && return 0.0
        -p * log(p) - (1 - p) * log(1 - p)
    end
    gaussian_left_probability_test(qcut; center=0.0, σ=1.0) =
        0.5 * (1 + erf_float_test((qcut - center) / σ))
    cat_left_probability_test(qcut; Q, σ=1.0) =
        0.5 * gaussian_left_probability_test(qcut; center=-Q, σ) +
        0.5 * gaussian_left_probability_test(qcut; center=Q, σ)

    @test binary_entropy_test(0.0) == 0.0
    @test binary_entropy_test(1.0) == 0.0
    @test binary_entropy_test(0.5) ≈ log(2) atol=1e-15
    @test exp(binary_entropy_test(0.5)) ≈ 2.0 atol=1e-15

    @test gaussian_left_probability_test(1.0e6; center=1.0e6) ≈ 0.5 atol=1e-15

    Q = 20.0
    @test cat_left_probability_test(-Q; Q) ≈ 0.25 atol=1e-14
    @test cat_left_probability_test(0.0; Q) ≈ 0.5 atol=1e-14
    @test cat_left_probability_test(Q; Q) ≈ 0.75 atol=1e-14
end

@testset "one-mode gates" begin
    qgrid = collect(range(-10.0, 10.0; length=1001))
    dq = qgrid[2] - qgrid[1]
    ψ = ComplexF64[π^(-1 / 4) * exp(-0.5 * q^2) for q in qgrid]
    normalize_grid_state!(qgrid, ψ)

    qmean(ϕ) = real(sum(conj.(ϕ) .* (qgrid .* ϕ)) * dq)
    q2mean(ϕ) = real(sum(abs2.(qgrid) .* abs2.(ϕ)) * dq)
    fidelity(ϕ, ξ) = abs2(grid_inner(qgrid, ϕ, ξ))

    @test grid_state_norm(qgrid, ψ) ≈ 1.0 atol=1e-10

    ψeval = linear_interpolating_eval(qgrid, ψ)
    @test ψeval(0.0) ≈ π^(-1 / 4) atol=1e-4
    @test ψeval(qgrid[1] - 1.0) == 0.0 + 0.0im

    ψx = apply_gate_to_grid(XDisplacementGate(1.0), qgrid, ψ)
    @test qmean(ψx) ≈ 1.0 atol=1e-3

    ψz = apply_gate_to_grid(ZDisplacementGate(0.7), qgrid, ψ)
    @test maximum(abs.(abs2.(ψz) .- abs2.(ψ))) < 1e-10

    ψw = apply_gate_to_grid(WeylDisplacementGate(1.0, 0.7), qgrid, ψ)
    @test qmean(ψw) ≈ 1.0 atol=1e-3

    ψquad = apply_gate_to_grid(QuadraticPhaseGate(0.25), qgrid, ψ)
    @test maximum(abs.(abs2.(ψquad) .- abs2.(ψ))) < 1e-10

    ψcubic = apply_gate_to_grid(CubicPhaseGate(0.05), qgrid, ψ)
    @test maximum(abs.(abs2.(ψcubic) .- abs2.(ψ))) < 1e-10

    r = 0.4
    ψs = apply_gate_to_grid(SqueezeGate(r), qgrid, ψ)
    @test q2mean(ψs) ≈ 0.5 * exp(-2r) atol=1e-3

    s = 0.3
    t = 0.7
    ψzx = apply_gate_to_grid(ZDisplacementGate(t), qgrid,
                             apply_gate_to_grid(XDisplacementGate(s), qgrid, ψ;
                                                renormalize=false))
    ψxz = apply_gate_to_grid(XDisplacementGate(s), qgrid,
                             apply_gate_to_grid(ZDisplacementGate(t), qgrid, ψ;
                                                renormalize=false))
    ψxz .*= exp(im * s * t)
    @test fidelity(ψzx, ψxz) > 1 - 1e-6

    phase_gates = (
        ZDisplacementGate(0.7),
        QuadraticPhaseGate(0.25),
        CubicPhaseGate(0.05),
    )
    for gate in phase_gates
        ψback = apply_gate_to_grid(inverse_gate(gate), qgrid,
                                   apply_gate_to_grid(gate, qgrid, ψ))
        @test fidelity(ψ, ψback) > 1 - 1e-10
    end

    ψxback = apply_gate_to_grid(inverse_gate(XDisplacementGate(0.3)), qgrid,
                                apply_gate_to_grid(XDisplacementGate(0.3), qgrid, ψ))
    @test fidelity(ψ, ψxback) > 1 - 1e-6

    ψwback = apply_gate_to_grid(inverse_gate(WeylDisplacementGate(0.3, 0.7)), qgrid,
                                apply_gate_to_grid(WeylDisplacementGate(0.3, 0.7),
                                                   qgrid, ψ))
    @test fidelity(ψ, ψwback) > 1 - 1e-6

    ψsback = apply_gate_to_grid(inverse_gate(SqueezeGate(0.2)), qgrid,
                                apply_gate_to_grid(SqueezeGate(0.2), qgrid, ψ))
    @test fidelity(ψ, ψsback) > 1 - 1e-5
end

@testset "one-mode GKP displacement errors" begin
    α = 2sqrt(pi)
    g = GridSpec(-8.0, 8.0, 201)
    qgrid = grid(g)
    Hdesc = regularized_gkp_hamiltonian(; ε=1.0, κ=0.05, α, boundary=:zero)
    fd = grid_eigenstates(Hdesc, g; nev=4)
    ψ0 = copy(fd.wavefunctions[:, 1])
    normalize_grid_state!(qgrid, ψ0)

    ψx_small = apply_gate_to_grid(XDisplacementGate(0.05), qgrid, ψ0)
    ψx_large = apply_gate_to_grid(XDisplacementGate(0.4), qgrid, ψ0)
    Fx_small = grid_subspace_overlap_abs2(qgrid, ψx_small, fd.wavefunctions; nstates=2)
    Fx_large = grid_subspace_overlap_abs2(qgrid, ψx_large, fd.wavefunctions; nstates=2)
    @test Fx_small > Fx_large
    @test Fx_small > 0.95

    ψz_small = apply_gate_to_grid(ZDisplacementGate(0.05), qgrid, ψ0)
    ψz_large = apply_gate_to_grid(ZDisplacementGate(0.4), qgrid, ψ0)
    Fz_small = grid_subspace_overlap_abs2(qgrid, ψz_small, fd.wavefunctions; nstates=2)
    Fz_large = grid_subspace_overlap_abs2(qgrid, ψz_large, fd.wavefunctions; nstates=2)
    @test Fz_small > Fz_large
    @test Fz_small > 0.95

    dx = gkp_diagnostics(qgrid, ψx_large; α, boundary=:zero)
    dz = gkp_diagnostics(qgrid, ψz_large; α, boundary=:zero)
    @test isfinite(dx.cosq)
    @test isfinite(dx.cosp)
    @test isfinite(dz.cosq)
    @test isfinite(dz.cosp)
end

@testset "GridMPS product states and one-site gates" begin
    qgrid = collect(range(-8.0, 8.0; length=201))
    ψ = ComplexF64[π^(-1 / 4) * exp(-0.5 * q^2) for q in qgrid]
    mps = product_gridmps(qgrid, (ψ, ψ))

    @test length(mps.sites) == 2
    @test size(mps.sites[1].A) == (1, length(qgrid), 1)
    @test gridmps_norm(mps) ≈ 1.0 atol=1e-10

    Ψ0 = gridmps_to_dense(mps)
    @test dense_q_mean(qgrid, Ψ0, 1) ≈ 0.0 atol=1e-10
    @test dense_q_mean(qgrid, Ψ0, 2) ≈ 0.0 atol=1e-10

    apply_one_mode_gate!(mps, 1, XDisplacementGate(1.0))
    Ψx = gridmps_to_dense(mps)
    @test gridmps_norm(mps) ≈ 1.0 atol=1e-10
    @test dense_q_mean(qgrid, Ψx, 1) ≈ 1.0 atol=1e-3
    @test dense_q_mean(qgrid, Ψx, 2) ≈ 0.0 atol=1e-10
end

@testset "GridMPS two-site block and SVD splitting" begin
    qgrid = collect(range(-6.0, 6.0; length=121))
    dq = qgrid[2] - qgrid[1]
    ψ = ComplexF64[π^(-1 / 4) * exp(-0.5 * q^2) for q in qgrid]
    mps = product_gridmps(qgrid, (ψ, ψ))
    Ψ0 = gridmps_to_dense(mps)

    Θ = two_site_block(mps, 1)
    Anew, Bnew, S, err = split_two_site_block(qgrid, Θ; χmax=4)
    @test length(S) == 4
    @test err ≈ 0.0 atol=1e-12
    @test S[1] ≈ 1.0 atol=1e-10
    @test truncation_error(S, 1) < 1e-12
    @test schmidt_entropy(S) < 1e-10

    replace_two_sites!(mps, 1, Anew, Bnew)
    Ψ1 = gridmps_to_dense(mps)
    @test real(sum(abs2, Ψ0 - Ψ1) * dq^2) < 1e-20
    @test gridmps_norm(mps) ≈ 1.0 atol=1e-10

    Θent = copy(Θ)
    for i in eachindex(qgrid), j in eachindex(qgrid)
        Θent[1, i, j, 1] *= exp(-0.25im * qgrid[i] * qgrid[j])
    end
    _, _, Sent, err1 = split_two_site_block(qgrid, Θent; χmax=1)
    _, _, Sent4, err4 = split_two_site_block(qgrid, Θent; χmax=4)
    @test schmidt_entropy(Sent4) > 0
    @test err4 < err1
    @test truncation_error(Sent4, length(Sent4)) ≈ 0.0 atol=1e-12
end

@testset "GridMPS cross-phase two-mode gate" begin
    qgrid = collect(range(-6.0, 6.0; length=121))
    dq = qgrid[2] - qgrid[1]
    ψ = ComplexF64[π^(-1 / 4) * exp(-0.5 * q^2) for q in qgrid]

    mps1 = product_gridmps(qgrid, (ψ, ψ))
    out1 = apply_two_mode_gate!(mps1, 1, CrossPhaseGate(0.35); χmax=1)

    mps4 = product_gridmps(qgrid, (ψ, ψ))
    out4 = apply_two_mode_gate!(mps4, 1, CrossPhaseGate(0.35); χmax=4)

    @test length(out4.singular_values) == 4
    @test schmidt_entropy(out4.singular_values) > 0.01
    @test out4.truncation_error < out1.truncation_error
    @test gridmps_norm(mps4) ≈ 1.0 atol=1e-10

    mps_inv = product_gridmps(qgrid, (ψ, ψ))
    Ψ0 = gridmps_to_dense(mps_inv)
    apply_two_mode_gate!(mps_inv, 1, CrossPhaseGate(0.25); χmax=16)
    apply_two_mode_gate!(mps_inv, 1, inverse_gate(CrossPhaseGate(0.25)); χmax=16)
    Ψback = gridmps_to_dense(mps_inv)
    @test real(sum(abs2, Ψ0 - Ψback) * dq^2) < 1e-10
end

@testset "GridMPS Gaussian two-mode gates" begin
    qgrid = collect(range(-6.0, 6.0; length=121))
    ψ = ComplexF64[π^(-1 / 4) * exp(-0.5 * q^2) for q in qgrid]

    mps_bs = product_gridmps(qgrid, (ψ, ψ))
    out_bs = apply_two_mode_gate!(mps_bs, 1, BeamSplitterGate(π / 4); χmax=6)
    @test gridmps_norm(mps_bs) ≈ 1.0 atol=1e-10
    @test schmidt_entropy(out_bs.singular_values) < 1e-4
    @test out_bs.singular_values[1] > 0.999

    mps_tms = product_gridmps(qgrid, (ψ, ψ))
    out_tms = apply_two_mode_gate!(mps_tms, 1, TwoModeSqueezerGate(0.4); χmax=8)
    @test gridmps_norm(mps_tms) ≈ 1.0 atol=1e-10
    @test schmidt_entropy(out_tms.singular_values) > 0.1
    @test out_tms.truncation_error < 1e-6
end

@testset "Fock projection diagnostics" begin
    qgrid = collect(range(-10.0, 10.0; length=1201))
    Φ = fock_basis_values(qgrid, 6)
    @test real(grid_inner(qgrid, Φ[:, 1], Φ[:, 1])) ≈ 1.0 atol=1e-10
    @test abs(grid_inner(qgrid, Φ[:, 1], Φ[:, 2])) < 1e-10

    ψvac = copy(Φ[:, 1])
    coeffs_vac = fock_coefficients_from_grid(qgrid, ψvac; Nmax=12)
    @test abs2(coeffs_vac[1]) ≈ 1.0 atol=1e-10
    @test fock_weight(coeffs_vac) ≈ 1.0 atol=1e-10
    @test required_fock_cutoff(coeffs_vac; tol=1e-8) == 1

    r = 0.7
    ψs = ComplexF64[π^(-1 / 4) * exp(r / 2) * exp(-0.5 * exp(2r) * q^2)
                    for q in qgrid]
    normalize_grid_state!(qgrid, ψs)
    d = gkp_diagnostics(qgrid, ψs)
    @test photon_number_from_qp(d.q2, d.p2) ≈ sinh(r)^2 atol=2e-3

    coeffs_s = fock_coefficients_from_grid(qgrid, ψs; Nmax=40)
    @test fock_weight(coeffs_s) > 1 - 1e-8
    @test required_fock_cutoff(coeffs_s; tol=1e-6) > 1
end
