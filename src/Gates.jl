export AbstractCVGate,
       AbstractOneModeGate,
       AbstractTwoModeGate,
       XDisplacementGate,
       ZDisplacementGate,
       WeylDisplacementGate,
       QuadraticPhaseGate,
       CubicPhaseGate,
       SqueezeGate,
       CrossPhaseGate,
       inverse_gate,
       apply_gate_to_function,
       apply_gate_to_grid,
       normalize_grid_state!,
       grid_state_norm,
       linear_interpolating_eval

abstract type AbstractCVGate end
abstract type AbstractOneModeGate <: AbstractCVGate end
abstract type AbstractTwoModeGate <: AbstractCVGate end

"""
    XDisplacementGate(s)

Position displacement `X(s) = exp(-i s p)` with convention
`(X(s)ψ)(q) = ψ(q - s)`.
"""
struct XDisplacementGate <: AbstractOneModeGate
    s::Float64
end

"""
    ZDisplacementGate(t)

Momentum displacement / phase ramp `Z(t) = exp(i t q)` with convention
`(Z(t)ψ)(q) = exp(i t q) ψ(q)`.
"""
struct ZDisplacementGate <: AbstractOneModeGate
    t::Float64
end

"""
    WeylDisplacementGate(s, t)

Combined displacement with symmetric Weyl phase:
`(D(s,t)ψ)(q) = exp(i t * (q - s/2)) ψ(q - s)`.
"""
struct WeylDisplacementGate <: AbstractOneModeGate
    s::Float64
    t::Float64
end

"""
    QuadraticPhaseGate(γ)

Gaussian quadratic phase gate: `ψ(q) -> exp(i γ q^2 / 2) ψ(q)`.
"""
struct QuadraticPhaseGate <: AbstractOneModeGate
    γ::Float64
end

"""
    CubicPhaseGate(γ)

Non-Gaussian cubic phase gate: `ψ(q) -> exp(i γ q^3) ψ(q)`.
"""
struct CubicPhaseGate <: AbstractOneModeGate
    γ::Float64
end

"""
    SqueezeGate(r)

Single-mode squeezing with convention `(S(r)ψ)(q) = exp(r/2) ψ(exp(r) q)`,
so that `S(r)' q S(r) = exp(-r) q`.
"""
struct SqueezeGate <: AbstractOneModeGate
    r::Float64
end

"""
    CrossPhaseGate(γ)

Two-mode diagonal q-basis gate:
`Ψ(q1,q2) -> exp(-i γ q1 q2) Ψ(q1,q2)`.
"""
struct CrossPhaseGate <: AbstractTwoModeGate
    γ::Float64
end

inverse_gate(g::XDisplacementGate) = XDisplacementGate(-g.s)
inverse_gate(g::ZDisplacementGate) = ZDisplacementGate(-g.t)
inverse_gate(g::WeylDisplacementGate) = WeylDisplacementGate(-g.s, -g.t)
inverse_gate(g::QuadraticPhaseGate) = QuadraticPhaseGate(-g.γ)
inverse_gate(g::CubicPhaseGate) = CubicPhaseGate(-g.γ)
inverse_gate(g::SqueezeGate) = SqueezeGate(-g.r)
inverse_gate(g::CrossPhaseGate) = CrossPhaseGate(-g.γ)

function apply_gate_to_function(g::XDisplacementGate, ψ)
    s = g.s
    return q -> ψ(q - s)
end

function apply_gate_to_function(g::ZDisplacementGate, ψ)
    t = g.t
    return q -> exp(im * t * q) * ψ(q)
end

function apply_gate_to_function(g::WeylDisplacementGate, ψ)
    s = g.s
    t = g.t
    return q -> exp(im * t * (q - s / 2)) * ψ(q - s)
end

function apply_gate_to_function(g::QuadraticPhaseGate, ψ)
    γ = g.γ
    return q -> exp(0.5im * γ * q^2) * ψ(q)
end

function apply_gate_to_function(g::CubicPhaseGate, ψ)
    γ = g.γ
    return q -> exp(im * γ * q^3) * ψ(q)
end

function apply_gate_to_function(g::SqueezeGate, ψ)
    a = exp(g.r)
    pref = sqrt(a)
    return q -> pref * ψ(a * q)
end

function grid_state_norm(qgrid::AbstractVector, ψgrid::AbstractVector)
    dq = grid_step(qgrid)
    return sqrt(real(sum(abs2, ψgrid) * dq))
end

function normalize_grid_state!(qgrid::AbstractVector, ψgrid::AbstractVector)
    nrm = grid_state_norm(qgrid, ψgrid)
    ψgrid ./= max(nrm, eps(Float64))
    return ψgrid
end

"""
    linear_interpolating_eval(qgrid, ψgrid; outside=0)

Return a function `ψ(q)` from linear interpolation of grid values. Values outside
the grid domain are set to `outside`.
"""
function linear_interpolating_eval(qgrid::AbstractVector, ψgrid::AbstractVector;
                                  outside::ComplexF64=0.0 + 0.0im)
    length(qgrid) == length(ψgrid) || error("qgrid and ψgrid length mismatch")
    qmin = qgrid[1]
    qmax = qgrid[end]
    dq = grid_step(qgrid)
    N = length(qgrid)
    ψvals = ComplexF64.(ψgrid)

    return function ψ(q)
        if q < qmin || q > qmax
            return outside
        end

        x = (q - qmin) / dq + 1
        i = floor(Int, x)
        w = x - i

        if i < 1
            return ψvals[1]
        elseif i >= N
            return ψvals[end]
        else
            return (1 - w) * ψvals[i] + w * ψvals[i + 1]
        end
    end
end

"""
    apply_gate_to_grid(g, qgrid, ψgrid; renormalize=true)

Apply a one-mode gate to a grid-sampled wavefunction by interpolation.
"""
function apply_gate_to_grid(g::AbstractOneModeGate, qgrid::AbstractVector,
                            ψgrid::AbstractVector; renormalize::Bool=true)
    ψeval = linear_interpolating_eval(qgrid, ψgrid)
    ψnew_eval = apply_gate_to_function(g, ψeval)
    ψnew = ComplexF64[ψnew_eval(q) for q in qgrid]

    if renormalize
        normalize_grid_state!(qgrid, ψnew)
    end

    return ψnew
end
