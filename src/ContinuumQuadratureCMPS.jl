module ContinuumQuadratureCMPS

using LinearAlgebra
using ITensors
using Optim
using Printf
using Random
using SparseArrays

export GridSpec, grid, trapz, CMPS, wavefunction, amplitude,
       normalize_values, Hamiltonian1D, energy, diagnostics,
       finite_difference_hamiltonian, grid_eigenstates,
       harmonic_hamiltonian, anharmonic_hamiltonian,
       ideal_gkp_hamiltonian, regularized_gkp_hamiltonian,
       translation_matrix, cos_p_matrix,
       ConstantGaugeCMPSFamily, nparams, random_constant_gauge_theta,
       unpack_constant_gauge, amplitude_constant_gauge,
       amplitudes_constant_gauge, normalized_amplitudes_constant_gauge,
       grid_inner, grid_expectation, grid_overlap_abs2,
       grid_subspace_overlap_abs2, grid_residual_norm,
       grid_boundary_weight, gkp_diagnostics,
       rayleigh_grid_energy, constant_gauge_energy,
       scalar_cmps, gaussian_cmps, quartic_trial_cmps,
       gkp_comb_cmps, optimize_scalar_params,
       itensor_amplitude_at

# -----------------------------------------------------------------------------
# Grid and quadrature helpers.
# This is only for numerical integration/evaluation. The variational object itself
# is a continuum function q -> A(q), B(q).
# -----------------------------------------------------------------------------

struct GridSpec
    qmin::Float64
    qmax::Float64
    N::Int
    function GridSpec(qmin::Real, qmax::Real, N::Integer)
        qmax > qmin || error("qmax must be larger than qmin")
        N >= 5 || error("N must be at least 5")
        new(Float64(qmin), Float64(qmax), Int(N))
    end
end

grid(g::GridSpec) = collect(range(g.qmin, g.qmax; length=g.N))

function trapz(qs::AbstractVector, fs::AbstractVector)
    length(qs) == length(fs) || error("trapz: qs and fs length mismatch")
    s = zero(promote_type(eltype(fs), eltype(qs)))
    @inbounds for i in 1:length(qs)-1
        h = qs[i+1] - qs[i]
        s += 0.5h * (fs[i] + fs[i+1])
    end
    return s
end

function central_derivative(qs::AbstractVector, ψ::AbstractVector)
    N = length(qs)
    dψ = similar(ψ)
    @inbounds begin
        dψ[1] = (ψ[2] - ψ[1]) / (qs[2] - qs[1])
        for i in 2:N-1
            dψ[i] = (ψ[i+1] - ψ[i-1]) / (qs[i+1] - qs[i-1])
        end
        dψ[N] = (ψ[N] - ψ[N-1]) / (qs[N] - qs[N-1])
    end
    return dψ
end

"""
    interp_shift(qs, ψ, shift; boundary=:zero)

Return q -> ψ(q + shift) sampled on `qs`, using linear interpolation.
Boundary choices:
- `:zero`: values outside `[minimum(qs), maximum(qs)]` are zero.
- `:periodic`: wrap into the interval before interpolating.
"""
function interp_shift(qs::AbstractVector{<:Real}, ψ::AbstractVector, shift::Real; boundary::Symbol=:zero)
    N = length(qs)
    qmin, qmax = qs[1], qs[end]
    h = qs[2] - qs[1]
    out = similar(ψ)
    L = qmax - qmin

    @inbounds for i in 1:N
        x = qs[i] + shift
        if boundary == :periodic
            # Map x into [qmin, qmax). Avoid the duplicated endpoint issue.
            x = qmin + mod(x - qmin, L)
        elseif boundary == :zero
            if x < qmin || x > qmax
                out[i] = zero(eltype(ψ))
                continue
            end
        else
            error("unknown boundary condition: $boundary")
        end

        # Clamp after possible periodic wrap.
        if x <= qmin
            out[i] = ψ[1]
        elseif x >= qmax
            out[i] = ψ[end]
        else
            t = (x - qmin) / h
            j = floor(Int, t) + 1
            j = clamp(j, 1, N-1)
            α = (x - qs[j]) / (qs[j+1] - qs[j])
            out[i] = (1 - α) * ψ[j] + α * ψ[j+1]
        end
    end
    return out
end

# -----------------------------------------------------------------------------
# Generic continuous matrix-product scalar wavefunction.
# -----------------------------------------------------------------------------

"""
    CMPS(χ, qmin, qmax, A, B, vL, vR)

A cMPS-like scalar wavefunction ansatz for one continuous quadrature coordinate:

    ψ(q) = vL' * U(qmin,q) * B(q) * U(q,qmax) * vR

where U(a,b) = P exp(∫_a^b A(s) ds). `A` and `B` are functions returning χ×χ
ComplexF64 matrices.

This is intentionally a single-particle / quadrature-space object, not the
standard many-boson field-theory cMPS with field creation operators.
"""
struct CMPS{FA,FB}
    χ::Int
    qmin::Float64
    qmax::Float64
    A::FA
    B::FB
    vL::Vector{ComplexF64}
    vR::Vector{ComplexF64}
end

function CMPS(χ::Integer, qmin::Real, qmax::Real, A::FA, B::FB,
              vL::AbstractVector, vR::AbstractVector) where {FA,FB}
    χi = Int(χ)
    length(vL) == χi || error("length(vL) must equal χ")
    length(vR) == χi || error("length(vR) must equal χ")
    return CMPS{FA,FB}(χi, Float64(qmin), Float64(qmax), A, B,
                       ComplexF64.(vL), ComplexF64.(vR))
end

function propagators(c::CMPS, qs::AbstractVector{<:Real})
    N = length(qs)
    χ = c.χ
    Uleft = Vector{Matrix{ComplexF64}}(undef, N)
    Uright = Vector{Matrix{ComplexF64}}(undef, N)
    Uleft[1] = Matrix{ComplexF64}(I, χ, χ)
    Uright[N] = Matrix{ComplexF64}(I, χ, χ)

    @inbounds for i in 1:N-1
        h = qs[i+1] - qs[i]
        mid = 0.5 * (qs[i+1] + qs[i])
        step = exp(h * c.A(mid))
        Uleft[i+1] = Uleft[i] * step
    end
    @inbounds for i in N-1:-1:1
        h = qs[i+1] - qs[i]
        mid = 0.5 * (qs[i+1] + qs[i])
        step = exp(h * c.A(mid))
        Uright[i] = step * Uright[i+1]
    end
    return Uleft, Uright
end

function propagation_grid_with(c::CMPS, q::Real, Nprop::Int)
    c.qmin <= q <= c.qmax || error("q must lie inside the CMPS interval")
    base = collect(range(c.qmin, c.qmax; length=Nprop))
    return unique(sort!(push!(base, Float64(q))))
end

function amplitude(c::CMPS, q::Real; Nprop::Int=800)
    qs = propagation_grid_with(c, q, Nprop)
    ψ = wavefunction(c, qs)
    i = argmin(abs.(qs .- Float64(q)))
    return ψ[i]
end

function wavefunction(c::CMPS, qs::AbstractVector{<:Real})
    Uleft, Uright = propagators(c, qs)
    ψ = Vector{ComplexF64}(undef, length(qs))
    @inbounds for i in eachindex(qs)
        ψ[i] = (c.vL' * Uleft[i] * c.B(qs[i]) * Uright[i] * c.vR)[1]
    end
    return ψ
end

function normalize_values(qs::AbstractVector, ψ::AbstractVector)
    nrm2 = real(trapz(qs, abs2.(ψ)))
    nrm2 > 0 || error("cannot normalize zero wavefunction")
    return ψ ./ sqrt(nrm2)
end

# Convenience scalar cMPS: χ=1, A(q)=0, B(q)=f(q). This is useful for exact
# Gaussian and controlled trial-function examples.
function scalar_cmps(f::Function, qmin::Real, qmax::Real)
    A(q) = fill(0.0 + 0.0im, 1, 1)
    B(q) = fill(ComplexF64(f(q)), 1, 1)
    return CMPS(1, qmin, qmax, A, B, [1.0 + 0im], [1.0 + 0im])
end

# -----------------------------------------------------------------------------
# ITensors bridge.
# -----------------------------------------------------------------------------

"""
    itensor_amplitude_at(c, q; Nprop=800)

Compute ψ(q) using an explicit ITensor contraction of the auxiliary legs. This is
mainly a sanity check / bridge: the high-throughput evaluator uses matrices
because matrix exponentials are simpler there.
"""
function itensor_amplitude_at(c::CMPS, q::Real; Nprop::Int=800)
    qs = propagation_grid_with(c, q, Nprop)
    Uleft, Uright = propagators(c, qs)
    i = argmin(abs.(qs .- Float64(q)))
    χ = c.χ

    l = Index(χ, "aux_l")
    a = Index(χ, "aux_a")
    b = Index(χ, "aux_b")
    r = Index(χ, "aux_r")

    VL = ITensor(conj(c.vL), l)
    UL = ITensor(Uleft[i], l, a)
    BT = ITensor(c.B(qs[i]), a, b)
    UR = ITensor(Uright[i], b, r)
    VR = ITensor(c.vR, r)

    T = VL * UL * BT * UR * VR
    return ComplexF64(T[])
end

# -----------------------------------------------------------------------------
# Hamiltonians in q representation.
# Hψ = -kinetic_coeff * ψ'' + V(q)ψ + sum_s c_s ψ(q+s)
# The expectation value uses integration by parts for kinetic energy:
# <ψ|-k∂²|ψ> = k ∫ |ψ'|², assuming tails/boundaries are under control.
# -----------------------------------------------------------------------------

struct Hamiltonian1D{FV}
    name::String
    kinetic_coeff::Float64
    V::FV
    translations::Vector{Tuple{ComplexF64,Float64}}
    boundary::Symbol
end

function energy(H::Hamiltonian1D, qs::AbstractVector, ψraw::AbstractVector)
    ψ = normalize_values(qs, ψraw)
    dψ = central_derivative(qs, ψ)

    kinetic = H.kinetic_coeff * real(trapz(qs, abs2.(dψ)))
    potential = real(trapz(qs, conj.(ψ) .* H.V.(qs) .* ψ))

    trans = 0.0
    for (c, s) in H.translations
        ψs = interp_shift(qs, ψ, s; boundary=H.boundary)
        trans += real(trapz(qs, conj.(ψ) .* (c .* ψs)))
    end
    return kinetic + potential + trans
end

function diagnostics(H::Hamiltonian1D, qs::AbstractVector, ψraw::AbstractVector; α::Float64=2sqrt(pi))
    ψ = normalize_values(qs, ψraw)
    dψ = central_derivative(qs, ψ)
    norm = real(trapz(qs, abs2.(ψ)))
    qmean = real(trapz(qs, qs .* abs2.(ψ)))
    q2 = real(trapz(qs, qs.^2 .* abs2.(ψ)))
    p2 = real(trapz(qs, abs2.(dψ)))
    cq = real(trapz(qs, cos.(α .* qs) .* abs2.(ψ)))
    ψp = interp_shift(qs, ψ, α; boundary=H.boundary)
    ψm = interp_shift(qs, ψ, -α; boundary=H.boundary)
    cp = real(0.5 * trapz(qs, conj.(ψ) .* (ψp .+ ψm)))
    return (; energy=energy(H, qs, ψ), norm, qmean, qvar=q2-qmean^2,
            p2, cos_αq=cq, cos_αp=cp)
end

function harmonic_hamiltonian(;ω::Real=1.0, boundary::Symbol=:zero)
    V(q) = 0.5 * Float64(ω)^2 * q^2
    return Hamiltonian1D("harmonic oscillator", 0.5, V, Tuple{ComplexF64,Float64}[], boundary)
end

function anharmonic_hamiltonian(;ω::Real=1.0, λ::Real=0.1, η::Real=0.0, boundary::Symbol=:zero)
    V(q) = 0.5 * Float64(ω)^2 * q^2 + Float64(λ) * q^4 + Float64(η) * q^6
    return Hamiltonian1D("anharmonic oscillator", 0.5, V, Tuple{ComplexF64,Float64}[], boundary)
end

function ideal_gkp_hamiltonian(;ε::Real=1.0, α::Real=2sqrt(pi), boundary::Symbol=:zero)
    αf, εf = Float64(α), Float64(ε)
    V(q) = -εf * cos(αf * q)
    trans = [(ComplexF64(-0.5εf), αf), (ComplexF64(-0.5εf), -αf)]
    return Hamiltonian1D("ideal GKP / Harper Hamiltonian", 0.0, V, trans, boundary)
end

function regularized_gkp_hamiltonian(;ε::Real=1.0, κ::Real=0.02, α::Real=2sqrt(pi), boundary::Symbol=:zero)
    αf, εf, κf = Float64(α), Float64(ε), Float64(κ)
    V(q) = -εf * cos(αf * q) + 0.5 * κf * q^2
    trans = [(ComplexF64(-0.5εf), αf), (ComplexF64(-0.5εf), -αf)]
    return Hamiltonian1D("regularized GKP Hamiltonian", 0.5κf, V, trans, boundary)
end

# -----------------------------------------------------------------------------
# Direct grid Hamiltonian baseline.
# -----------------------------------------------------------------------------

function shift_interpolation_matrix(qs::AbstractVector{<:Real}, shift::Real; boundary::Symbol=:zero)
    N = length(qs)
    qmin, qmax = qs[1], qs[end]
    h = qs[2] - qs[1]
    L = qmax - qmin
    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]

    @inbounds for i in 1:N
        x = qs[i] + shift
        if boundary == :periodic
            x = qmin + mod(x - qmin, L)
        elseif boundary == :zero
            if x < qmin || x > qmax
                continue
            end
        else
            error("unknown boundary condition: $boundary")
        end

        if x <= qmin
            push!(rows, i); push!(cols, 1); push!(vals, 1 + 0im)
        elseif x >= qmax
            push!(rows, i); push!(cols, N); push!(vals, 1 + 0im)
        else
            t = (x - qmin) / h
            j = clamp(floor(Int, t) + 1, 1, N - 1)
            α = (x - qs[j]) / (qs[j+1] - qs[j])
            push!(rows, i); push!(cols, j); push!(vals, ComplexF64(1 - α))
            push!(rows, i); push!(cols, j + 1); push!(vals, ComplexF64(α))
        end
    end
    return sparse(rows, cols, vals, N, N)
end

"""
    translation_matrix(qgrid, shift; boundary=:zero)

Sparse interpolation matrix representing `ψ(q + shift)` on `qgrid`.
With `boundary=:zero`, samples shifted outside the interval are discarded.
"""
translation_matrix(qs::AbstractVector{<:Real}, shift::Real; boundary::Symbol=:zero) =
    shift_interpolation_matrix(qs, shift; boundary)

"""
    cos_p_matrix(qgrid, α; boundary=:zero)

Grid representation of `cos(αp)`, using
`cos(αp)ψ(q) = (ψ(q+α) + ψ(q-α))/2`.
The finite-grid matrix is explicitly symmetrized because interpolation and
boundaries otherwise break Hermiticity at machine precision.
"""
function cos_p_matrix(qs::AbstractVector{<:Real}, α::Real; boundary::Symbol=:zero)
    C = 0.5 .* (translation_matrix(qs, α; boundary) +
                translation_matrix(qs, -α; boundary))
    return 0.5 .* (C .+ C')
end

"""
    finite_difference_hamiltonian(H, grid_spec)

Build a sparse grid Hamiltonian for `Hamiltonian1D` using the same
Dirichlet-ish second derivative and interpolation conventions as the evaluator.
This is the direct quadrature-grid baseline for checking variational ansaetze.
The returned matrix is the Hermitian finite-grid representative of the continuum
Hamiltonian.
"""
function finite_difference_hamiltonian(H::Hamiltonian1D, g::GridSpec)
    qs = grid(g)
    N = length(qs)
    h = qs[2] - qs[1]
    main = fill(-2.0, N)
    off = fill(1.0, N - 1)
    D2 = spdiagm(-1 => off, 0 => main, 1 => off) / h^2
    M = ComplexF64.(-H.kinetic_coeff .* D2 + spdiagm(0 => H.V.(qs)))

    for (c, s) in H.translations
        M += c .* shift_interpolation_matrix(qs, s; boundary=H.boundary)
    end
    return 0.5 .* (M .+ M'), qs
end

"""
    grid_eigenstates(H, grid_spec; nev=6)

Dense diagonalization wrapper for small-to-medium grid baselines. Eigenvectors
are normalized with the same uniform quadrature convention as `grid_inner`.
"""
function grid_eigenstates(H::Hamiltonian1D, g::GridSpec; nev::Int=6)
    M, qs = finite_difference_hamiltonian(H, g)
    nev <= g.N || error("nev must be <= grid size")
    F = eigen(Hermitian(Matrix(M)))
    vals = F.values[1:nev]
    vecs = ComplexF64.(F.vectors[:, 1:nev])
    for k in 1:nev
        vecs[:, k] ./= sqrt(real(grid_inner(qs, vecs[:, k], vecs[:, k])))
    end
    return (; q=qs, energies=vals, wavefunctions=vecs, matrix=M)
end

# -----------------------------------------------------------------------------
# First controlled χ > 1 family.
# -----------------------------------------------------------------------------

"""
    ConstantGaugeCMPSFamily(χ, qmin, qmax)

A first nontrivial quadrature-space continuous MPS family:

    ψ(q) = exp(-γq^2) * vL' * exp(A*(q-qmin)) * B * exp(A*(qmax-q)) * vR

with the stability-oriented parameterization

    A = -0.5*C'*C - im*K,  K = K'

followed by removal of the scalar trace drift. This is not a complete quotient
of gauge redundancy, but it gives a controlled matrix-valued ansatz before
moving to fully q-dependent A(q), B(q).
"""
struct ConstantGaugeCMPSFamily
    χ::Int
    qmin::Float64
    qmax::Float64
    function ConstantGaugeCMPSFamily(χ::Integer, qmin::Real, qmax::Real)
        χ >= 1 || error("χ must be positive")
        qmax > qmin || error("qmax must be larger than qmin")
        new(Int(χ), Float64(qmin), Float64(qmax))
    end
end

nparams(fam::ConstantGaugeCMPSFamily) = 6 * fam.χ^2 + 4 * fam.χ + 1

function hermitian_from_reals(xre::AbstractMatrix, xim::AbstractMatrix)
    X = ComplexF64.(xre) .+ im .* ComplexF64.(xim)
    return 0.5 .* (X + X')
end

function unpack_constant_gauge(theta::AbstractVector, fam::ConstantGaugeCMPSFamily)
    χ = fam.χ
    length(theta) == nparams(fam) || error("theta length must equal nparams(fam)")
    n = χ^2
    idx = 1

    K_re = reshape(Float64.(theta[idx:idx+n-1]), χ, χ)
    idx += n
    K_im = reshape(Float64.(theta[idx:idx+n-1]), χ, χ)
    idx += n

    C_re = reshape(Float64.(theta[idx:idx+n-1]), χ, χ)
    idx += n
    C_im = reshape(Float64.(theta[idx:idx+n-1]), χ, χ)
    idx += n

    B_re = reshape(Float64.(theta[idx:idx+n-1]), χ, χ)
    idx += n
    B_im = reshape(Float64.(theta[idx:idx+n-1]), χ, χ)
    idx += n

    vL_re = Float64.(theta[idx:idx+χ-1])
    idx += χ
    vL_im = Float64.(theta[idx:idx+χ-1])
    idx += χ

    vR_re = Float64.(theta[idx:idx+χ-1])
    idx += χ
    vR_im = Float64.(theta[idx:idx+χ-1])
    idx += χ

    logγ = Float64(theta[idx])

    K = hermitian_from_reals(K_re, K_im)
    C = ComplexF64.(C_re) .+ im .* ComplexF64.(C_im)
    B = ComplexF64.(B_re) .+ im .* ComplexF64.(B_im)

    vL = ComplexF64.(vL_re) .+ im .* ComplexF64.(vL_im)
    vR = ComplexF64.(vR_re) .+ im .* ComplexF64.(vR_im)
    vL ./= max(norm(vL), eps(Float64))
    vR ./= max(norm(vR), eps(Float64))

    γ = exp(logγ)
    A = -0.5 .* (C' * C) .- im .* K
    A -= (tr(A) / χ) .* Matrix{ComplexF64}(I, χ, χ)

    return A, B, vL, vR, γ
end

function random_constant_gauge_theta(fam::ConstantGaugeCMPSFamily; scale::Real=0.05,
                                     gamma::Real=0.5, rng=Random.default_rng())
    θ = Float64(scale) .* randn(rng, nparams(fam))
    θ[end] = log(Float64(gamma))
    return θ
end

function amplitude_constant_gauge(q::Real, theta::AbstractVector, fam::ConstantGaugeCMPSFamily)
    fam.qmin <= q <= fam.qmax || error("q must lie inside the family interval")
    A, B, vL, vR, γ = unpack_constant_gauge(theta, fam)
    UL = exp(A * (Float64(q) - fam.qmin))
    UR = exp(A * (fam.qmax - Float64(q)))
    envelope = exp(-γ * Float64(q)^2)
    return envelope * dot(vL, UL * B * UR * vR)
end

function amplitudes_constant_gauge(qgrid::AbstractVector, theta::AbstractVector,
                                   fam::ConstantGaugeCMPSFamily)
    return ComplexF64[amplitude_constant_gauge(q, theta, fam) for q in qgrid]
end

function grid_step(qgrid::AbstractVector{<:Real})
    length(qgrid) >= 2 || error("qgrid must contain at least two points")
    dq = qgrid[2] - qgrid[1]
    for i in 2:length(qgrid)-1
        isapprox(qgrid[i+1] - qgrid[i], dq; rtol=1e-10, atol=1e-12) ||
            error("qgrid must be uniformly spaced")
    end
    return Float64(dq)
end

function normalized_amplitudes_constant_gauge(qgrid::AbstractVector, theta::AbstractVector,
                                              fam::ConstantGaugeCMPSFamily)
    ψ = amplitudes_constant_gauge(qgrid, theta, fam)
    dq = grid_step(qgrid)
    nrm = sqrt(real(sum(abs2, ψ) * dq))
    nrm > 0 || error("cannot normalize zero wavefunction")
    return ψ ./ nrm
end

function grid_inner(qgrid::AbstractVector, ψ::AbstractVector, ϕ::AbstractVector)
    length(qgrid) == length(ψ) == length(ϕ) || error("qgrid, ψ, and ϕ length mismatch")
    dq = grid_step(qgrid)
    return sum(conj.(ψ) .* ϕ) * dq
end

function grid_overlap_abs2(qgrid::AbstractVector, ψ::AbstractVector, ϕ::AbstractVector)
    return abs2(grid_inner(qgrid, ψ, ϕ))
end

"""
    grid_subspace_overlap_abs2(qgrid, ψ, Φ; nstates)

Return the squared projection of `ψ` onto the first `nstates` columns of `Φ`,
with all inner products evaluated using the grid quadrature rule.
"""
function grid_subspace_overlap_abs2(qgrid::AbstractVector, ψ::AbstractVector,
                                    Φ::AbstractMatrix; nstates::Int)
    length(qgrid) == length(ψ) || error("qgrid and ψ length mismatch")
    size(Φ, 1) == length(ψ) || error("subspace basis row count mismatch")
    1 <= nstates <= size(Φ, 2) || error("nstates must be between 1 and the basis width")

    total = 0.0
    for k in 1:nstates
        ϕk = @view Φ[:, k]
        total += abs2(grid_inner(qgrid, ϕk, ψ))
    end
    return real(total)
end

function grid_residual_norm(qgrid::AbstractVector, H, ψ::AbstractVector, E::Real)
    length(qgrid) == length(ψ) || error("qgrid and ψ length mismatch")
    size(H, 1) == length(ψ) && size(H, 2) == length(ψ) || error("H size mismatch")
    dq = grid_step(qgrid)
    r = H * ψ .- Float64(E) .* ψ
    return sqrt(real(sum(abs2, r) * dq))
end

function grid_expectation(qgrid::AbstractVector, Op, ψ::AbstractVector)
    length(qgrid) == length(ψ) || error("qgrid and ψ length mismatch")
    size(Op, 1) == length(ψ) && size(Op, 2) == length(ψ) || error("operator size mismatch")
    return real(grid_inner(qgrid, ψ, Op * ψ))
end

function grid_boundary_weight(qgrid::AbstractVector, ψ::AbstractVector; nedge::Int=10)
    length(qgrid) == length(ψ) || error("qgrid and ψ length mismatch")
    nedge > 0 || error("nedge must be positive")
    n = min(nedge, length(ψ) ÷ 2)
    dq = grid_step(qgrid)
    return real((sum(abs2, @view ψ[1:n]) + sum(abs2, @view ψ[end-n+1:end])) * dq)
end

function p2_matrix(qgrid::AbstractVector{<:Real})
    g = GridSpec(qgrid[1], qgrid[end], length(qgrid))
    Hfree = Hamiltonian1D("free kinetic", 0.5, q -> 0.0, Tuple{ComplexF64,Float64}[], :zero)
    P2_over_2, _ = finite_difference_hamiltonian(Hfree, g)
    return 2 .* P2_over_2
end

function gkp_diagnostics(qgrid::AbstractVector, ψ::AbstractVector; α::Real=2sqrt(pi),
                         boundary::Symbol=:zero, nedge::Int=10)
    ψn = ψ ./ sqrt(real(grid_inner(qgrid, ψ, ψ)))
    cosq = spdiagm(0 => cos.(Float64(α) .* qgrid))
    cosp = cos_p_matrix(qgrid, α; boundary)
    q2 = spdiagm(0 => qgrid.^2)
    p2 = p2_matrix(qgrid)
    return (; norm=real(grid_inner(qgrid, ψn, ψn)),
            cosq=grid_expectation(qgrid, cosq, ψn),
            cosp=grid_expectation(qgrid, cosp, ψn),
            q2=grid_expectation(qgrid, q2, ψn),
            p2=grid_expectation(qgrid, p2, ψn),
            boundary_weight=grid_boundary_weight(qgrid, ψn; nedge))
end

function rayleigh_grid_energy(qgrid::AbstractVector, H, ψ::AbstractVector)
    length(qgrid) == length(ψ) || error("qgrid and ψ length mismatch")
    size(H, 1) == length(ψ) && size(H, 2) == length(ψ) || error("H size mismatch")
    numerator = real(grid_inner(qgrid, ψ, H * ψ))
    denominator = real(grid_inner(qgrid, ψ, ψ))
    denominator > 0 || error("zero wavefunction has no Rayleigh quotient")
    return numerator / denominator
end

function constant_gauge_energy(theta::AbstractVector, fam::ConstantGaugeCMPSFamily,
                               qgrid::AbstractVector, H)
    ψ = normalized_amplitudes_constant_gauge(qgrid, theta, fam)
    E = rayleigh_grid_energy(qgrid, H, ψ)
    return isfinite(E) ? E : 1e100
end

# -----------------------------------------------------------------------------
# Example ansatz builders.
# -----------------------------------------------------------------------------

gaussian_cmps(;ω::Real=1.0, qmin::Real=-8, qmax::Real=8) =
    scalar_cmps(q -> (Float64(ω)/pi)^0.25 * exp(-0.5 * Float64(ω) * q^2), qmin, qmax)

function quartic_trial_cmps(a::Real, b::Real; qmin::Real=-8, qmax::Real=8)
    af, bf = Float64(a), Float64(b)
    scalar_cmps(q -> exp(-af * q^2 - bf * q^4), qmin, qmax)
end

"""
    gkp_comb_cmps(; Δ, envelope, logical, nmax, spacing, qmin, qmax)

Finite-energy scalar trial comb. `logical=0` gives peaks at 2n*sqrt(pi),
`logical=1` shifts by sqrt(pi). This is not the exact GKP state, which is
non-normalizable; it is an approximate envelope-regularized comb.
"""
function gkp_comb_cmps(;Δ::Real=0.25, envelope::Real=0.03, logical::Int=0,
                       nmax::Int=10, spacing::Real=sqrt(pi),
                       qmin::Real=-12, qmax::Real=12)
    Δf = Float64(Δ)
    env = Float64(envelope)
    sp = Float64(spacing)
    shift = logical == 0 ? 0.0 : sp
    centers = [2n * sp + shift for n in -nmax:nmax]
    function f(q)
        s = 0.0
        for c in centers
            s += exp(-0.5 * ((q - c) / Δf)^2)
        end
        return exp(-0.5 * env * q^2) * s
    end
    return scalar_cmps(f, qmin, qmax)
end

# -----------------------------------------------------------------------------
# Small black-box optimizer for scalar parameterized trial cMPS families.
# For serious cMPS optimization, replace this with gauge-aware AD/custom gradients.
# -----------------------------------------------------------------------------

softplus(x) = log1p(exp(-abs(x))) + max(x, 0)

"""
    optimize_scalar_params(build, θ0, H, grid; iterations=1000)

`build(θ)` must return a CMPS. This uses derivative-free Nelder-Mead because it
survives non-smooth exploratory ansätze and avoids differentiating through expm.
"""
function optimize_scalar_params(build::Function, θ0::AbstractVector, H::Hamiltonian1D, g::GridSpec;
                                iterations::Int=1000, show_trace::Bool=true)
    qs = grid(g)
    function objective(θ)
        c = build(θ)
        ψ = wavefunction(c, qs)
        E = energy(H, qs, ψ)
        return isfinite(E) ? E : 1e20
    end
    res = optimize(objective, Float64.(θ0), NelderMead(),
                   Optim.Options(iterations=iterations, show_trace=show_trace))
    θopt = Optim.minimizer(res)
    copt = build(θopt)
    ψopt = normalize_values(qs, wavefunction(copt, qs))
    return (; result=res, theta=θopt, cmps=copt, qs, psi=ψopt,
            energy=energy(H, qs, ψopt), diagnostics=diagnostics(H, qs, ψopt))
end

end # module
