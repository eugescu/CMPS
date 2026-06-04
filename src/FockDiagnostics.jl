export fock_basis_values,
       fock_coefficients_from_grid,
       fock_cumulative_weight,
       fock_weight,
       required_fock_cutoff,
       photon_number_from_qp

"""
    fock_basis_values(qgrid, Nmax)

Return harmonic-oscillator basis functions ``ϕ_n(q)`` for `n = 0:Nmax-1`,
sampled on `qgrid`. The recurrence uses the normalized physicists' Hermite
functions with oscillator convention `H = (p^2 + q^2)/2`.
"""
function fock_basis_values(qgrid::AbstractVector, Nmax::Integer)
    Nmax > 0 || error("Nmax must be positive")
    qs = Float64.(qgrid)
    Φ = Matrix{ComplexF64}(undef, length(qs), Int(Nmax))
    Φ[:, 1] .= ComplexF64.(pi^(-1 / 4) .* exp.(-0.5 .* qs.^2))

    if Nmax >= 2
        Φ[:, 2] .= sqrt(2.0) .* qs .* Φ[:, 1]
    end
    for n in 1:Nmax-2
        Φ[:, n + 2] .= sqrt(2 / (n + 1)) .* qs .* Φ[:, n + 1] .-
                       sqrt(n / (n + 1)) .* Φ[:, n]
    end
    return Φ
end

function fock_coefficients_from_grid(qgrid::AbstractVector, ψ::AbstractVector; Nmax::Integer)
    length(qgrid) == length(ψ) || error("qgrid and ψ length mismatch")
    Φ = fock_basis_values(qgrid, Nmax)
    coeffs = ComplexF64[]
    for n in 1:Nmax
        push!(coeffs, grid_inner(qgrid, @view(Φ[:, n]), ψ))
    end
    return coeffs
end

fock_weight(coeffs::AbstractVector) = real(sum(abs2, coeffs))

function fock_cumulative_weight(coeffs::AbstractVector)
    out = zeros(Float64, length(coeffs))
    acc = 0.0
    for i in eachindex(coeffs)
        acc += abs2(coeffs[i])
        out[i] = real(acc)
    end
    return out
end

function required_fock_cutoff(coeffs::AbstractVector; tol::Real=1e-6)
    cumulative = fock_cumulative_weight(coeffs)
    target = (1 - Float64(tol)) * max(cumulative[end], eps(Float64))
    for i in eachindex(cumulative)
        if cumulative[i] >= target
            return i
        end
    end
    return length(coeffs)
end

photon_number_from_qp(q2::Real, p2::Real) = 0.5 * (Float64(q2) + Float64(p2) - 1)
