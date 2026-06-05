# Example 19: large momentum displacement in q-space.
#
# A large q displacement is a range problem. A large p displacement is a
# q-space resolution problem: Z(P) psi(q) = exp(i P q) psi(q). The density is
# unchanged, but the real/imaginary parts oscillate with wavelength 2pi/P.
#
# Run:
#   julia --project=. examples/19_large_momentum_displacement_qspace.jl
#
# Optional artifacts:
#   CMPS_WRITE_DATA=1 julia --project=. examples/19_large_momentum_displacement_qspace.jl
#   CMPS_WRITE_DATA=1 CMPS_PLOTS=1 julia --project=. examples/19_large_momentum_displacement_qspace.jl

using Printf
include("../src/ContinuumQuadratureCMPS.jl")
using .ContinuumQuadratureCMPS
include("example_io.jl")

function momentum_displaced_vacuum(qgrid, P)
    ψ = ComplexF64[pi^(-1 / 4) * exp(-0.5 * q^2) * exp(1im * P * q)
                  for q in qgrid]
    nrm = sqrt(real(trapz(qgrid, abs2.(ψ))))
    ψ ./= nrm
    return ψ
end

phase_wavelength(P) = P == 0 ? Inf : 2π / abs(P)

function grid_points_proxy(P; qmin=-8.0, qmax=8.0, base_N=801, phase_resolution=0.1)
    P == 0 && return base_N
    return ceil(Int, (qmax - qmin) * abs(P) / phase_resolution) + 1
end

fock_proxy(P) = max(5, ceil(Int, 0.5 * P^2))

function row(P)
    return (; P,
            wavelength=phase_wavelength(P),
            grid_N=grid_points_proxy(P),
            fock=fock_proxy(P),
            params=2)
end

@printf("large momentum displacement in q-space\n")
@printf("density is unchanged, but Re ψ(q) and phase oscillate with wavelength 2π/P\n")
@printf("%11s %17s %14s %14s %8s\n",
        "P", "phase wavelength", "grid N proxy", "Fock proxy", "params")

Ps = (0.0, 1.0e2, 1.0e3, 1.0e6)
rows = [row(P) for P in Ps]
for r in rows
    @printf("%11.3e %17.6e %14d %14d %8d\n",
            r.P, r.wavelength, r.grid_N, r.fock, r.params)
end

if get(ENV, "CMPS_WRITE_DATA", "0") == "1"
    scaling_path = write_momentum_scaling_csv(
        joinpath("outputs", "momentum_displacement_scaling.csv"),
        [r.P for r in rows],
        [r.wavelength for r in rows],
        [r.grid_N for r in rows],
        [r.fock for r in rows],
        [r.params for r in rows],
    )
    println("wrote scaling CSV: ", scaling_path)

    qgrid = collect(range(-8.0, 8.0; length=8001))
    cases = []

    for Pviz in (20.0, 50.0)
        ψ = momentum_displaced_vacuum(qgrid, Pviz)
        phase = Pviz .* qgrid
        label = "P=$(round(Int, Pviz))"

        phase_path = write_phase_csv(
            joinpath("outputs", "momentum_displacement_P$(round(Int, Pviz))_phase.csv"),
            qgrid,
            ψ,
            phase,
        )
        println("wrote phase CSV: ", phase_path)
        push!(cases, (; qgrid, psi=ψ, phase, label))
    end

    plot_path = maybe_plot_wavefunction_phase_comparison(
        joinpath("outputs", "momentum_displacement_P20_P50_real_phase.svg"),
        cases;
        title="Momentum displacement: P=20 and P=50 phase ramps",
    )
    if plot_path !== nothing
        println("wrote phase plot: ", plot_path)
    end
end
