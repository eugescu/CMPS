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
