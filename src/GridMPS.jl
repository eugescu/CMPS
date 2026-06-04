export GridMPSSite,
       GridMPS,
       product_gridmps,
       gridmps_to_dense,
       gridmps_norm,
       apply_one_mode_gate!,
       dense_q_mean

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
