export GridMPSSite,
       GridMPS,
       product_gridmps,
       gridmps_to_dense,
       gridmps_norm,
       apply_one_mode_gate!,
       dense_q_mean,
       two_site_block,
       split_two_site_block,
       replace_two_sites!,
       schmidt_entropy,
       truncation_error

struct GridMPSSite
    A::Array{ComplexF64,3}
    function GridMPSSite(A::Array{ComplexF64,3})
        size(A, 2) > 0 || error("site must have at least one q-grid sample")
        new(A)
    end
end

struct GridMPS
    qgrid::Vector{Float64}
    sites::Vector{GridMPSSite}
    function GridMPS(qgrid::AbstractVector{<:Real}, sites::Vector{GridMPSSite})
        length(qgrid) >= 2 || error("qgrid must contain at least two points")
        isempty(sites) && error("GridMPS must contain at least one site")
        N = length(qgrid)
        for site in sites
            size(site.A, 2) == N || error("all sites must use the qgrid length")
        end
        new(Float64.(qgrid), sites)
    end
end

function product_gridmps(qgrid::AbstractVector{<:Real}, ψs)
    sites = GridMPSSite[]
    for ψraw in ψs
        ψ = ComplexF64.(ψraw)
        length(ψ) == length(qgrid) || error("product state length mismatch")
        normalize_grid_state!(qgrid, ψ)
        A = reshape(ψ, 1, length(qgrid), 1)
        push!(sites, GridMPSSite(copy(A)))
    end
    return GridMPS(qgrid, sites)
end

function gridmps_to_dense(mps::GridMPS)
    Nsites = length(mps.sites)
    Nq = length(mps.qgrid)
    dims = ntuple(_ -> Nq, Nsites)
    Ψ = Array{ComplexF64}(undef, dims)

    for I in CartesianIndices(Ψ)
        M = reshape(mps.sites[1].A[1, I[1], :], 1, size(mps.sites[1].A, 3))
        for site in 2:Nsites
            A = mps.sites[site].A
            M = M * reshape(A[:, I[site], :], size(A, 1), size(A, 3))
        end
        Ψ[I] = M[1, 1]
    end

    return Ψ
end

function gridmps_norm(mps::GridMPS)
    Ψ = gridmps_to_dense(mps)
    dq = grid_step(mps.qgrid)
    return sqrt(real(sum(abs2, Ψ) * dq^length(mps.sites)))
end

function apply_one_mode_gate!(mps::GridMPS, site::Integer, gate::AbstractOneModeGate;
                              renormalize::Bool=true)
    1 <= site <= length(mps.sites) || error("site index out of bounds")
    A = mps.sites[site].A
    χL, _, χR = size(A)
    Anew = similar(A)

    for α in 1:χL, β in 1:χR
        ψ = vec(A[α, :, β])
        Anew[α, :, β] .= apply_gate_to_grid(gate, mps.qgrid, ψ; renormalize=false)
    end

    mps.sites[site] = GridMPSSite(Anew)
    if renormalize
        Ψnorm = gridmps_norm(mps)
        mps.sites[1].A ./= max(Ψnorm, eps(Float64))
    end
    return mps
end

function dense_q_mean(qgrid::AbstractVector, Ψ, site::Integer)
    1 <= site <= ndims(Ψ) || error("site index out of bounds")
    dq = grid_step(qgrid)
    total = 0.0
    for I in CartesianIndices(Ψ)
        total += qgrid[I[site]] * abs2(Ψ[I])
    end
    return real(total * dq^ndims(Ψ))
end

function two_site_block(mps::GridMPS, i::Integer)
    1 <= i < length(mps.sites) || error("site index must select adjacent pair")
    A = mps.sites[i].A
    B = mps.sites[i + 1].A
    χL, N, χM = size(A)
    χM2, N2, χR = size(B)
    χM == χM2 || error("neighboring bond dimensions do not match")
    N == N2 || error("neighboring site q-grid dimensions do not match")

    Θ = zeros(ComplexF64, χL, N, N, χR)
    for α in 1:χL, iq in 1:N, jq in 1:N, β in 1:χR
        acc = zero(ComplexF64)
        for a in 1:χM
            acc += A[α, iq, a] * B[a, jq, β]
        end
        Θ[α, iq, jq, β] = acc
    end
    return Θ
end

function truncation_error(S::AbstractVector, χkeep::Integer)
    0 <= χkeep <= length(S) || error("χkeep must lie between 0 and length(S)")
    p = abs2.(S)
    return real(sum(@view p[χkeep+1:end]))
end

function schmidt_entropy(S::AbstractVector)
    p = abs2.(S)
    total = sum(p)
    total > 0 || return 0.0
    p ./= total
    return real(-sum(pk -> pk > 0 ? pk * log(pk) : 0.0, p))
end

function choose_bond_dimension(S::AbstractVector; χmax::Int, cutoff::Real)
    χcap = min(χmax, length(S))
    cutoff <= 0 && return χcap

    p = abs2.(S)
    total = sum(p)
    for χ in 1:χcap
        if sum(@view p[χ+1:end]) <= Float64(cutoff) * total
            return χ
        end
    end
    return χcap
end

function split_two_site_block(qgrid::AbstractVector, Θ; χmax::Int=typemax(Int),
                              cutoff::Real=0.0)
    ndims(Θ) == 4 || error("two-site block must have four dimensions")
    χL, N1, N2, χR = size(Θ)
    length(qgrid) == N1 == N2 || error("qgrid length must match both physical legs")
    dq = grid_step(qgrid)

    Θw = dq .* Θ
    M = reshape(Θw, χL * N1, N2 * χR)
    F = svd(M)
    χ = choose_bond_dimension(F.S; χmax, cutoff)

    U = F.U[:, 1:χ]
    S = F.S[1:χ]
    Vt = F.Vt[1:χ, :]
    Anew = reshape(U, χL, N1, χ) ./ sqrt(dq)
    Bnew = reshape(Diagonal(S) * Vt, χ, N2, χR) ./ sqrt(dq)
    err = truncation_error(F.S, χ)

    return Anew, Bnew, S, err
end

function replace_two_sites!(mps::GridMPS, i::Integer, Anew, Bnew)
    1 <= i < length(mps.sites) || error("site index must select adjacent pair")
    size(Anew, 2) == length(mps.qgrid) || error("left replacement q dimension mismatch")
    size(Bnew, 2) == length(mps.qgrid) || error("right replacement q dimension mismatch")
    size(Anew, 3) == size(Bnew, 1) || error("replacement bond dimensions mismatch")
    if i > 1
        size(mps.sites[i - 1].A, 3) == size(Anew, 1) ||
            error("left external bond dimension mismatch")
    end
    if i + 1 < length(mps.sites)
        size(Bnew, 3) == size(mps.sites[i + 2].A, 1) ||
            error("right external bond dimension mismatch")
    end

    mps.sites[i] = GridMPSSite(ComplexF64.(Anew))
    mps.sites[i + 1] = GridMPSSite(ComplexF64.(Bnew))
    return mps
end
