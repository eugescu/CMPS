function write_density_csv(path, qgrid, ψ; center=0.0)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "x,q,density,realpsi,imagpsi")
        for i in eachindex(qgrid)
            x = qgrid[i] - center
            ρ = abs2(ψ[i])
            println(io, "$(x),$(qgrid[i]),$(ρ),$(real(ψ[i])),$(imag(ψ[i]))")
        end
    end
    return path
end

function maybe_plot_density(path, qgrid, ψ; center=0.0, title="")
    if get(ENV, "CMPS_PLOTS", "0") != "1"
        return nothing
    end

    try
        @eval import Plots

        x = qgrid .- center
        ρ = abs2.(ψ)

        p = Base.invokelatest(
            Plots.plot,
            x,
            ρ;
            xlabel="q - center",
            ylabel="|ψ(q)|²",
            title=title,
            legend=false,
        )

        mkpath(dirname(path))

        Base.invokelatest(Plots.savefig, p, path)

        return path
    catch err
        @warn "CMPS_PLOTS=1 was set, but plotting failed. Inspect CSV output instead." exception=(err, catch_backtrace())
        return nothing
    end
end

function write_scaling_csv(path, Qs, gridNs, fockProxies, localizedParams)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "Q,separation,uniform_grid_N,fock_proxy,localized_params")
        for i in eachindex(Qs)
            Q = Qs[i]
            println(io, "$(Q),$(2Q),$(gridNs[i]),$(fockProxies[i]),$(localizedParams[i])")
        end
    end
    return path
end

function write_entropy_csv(path, qcuts, local_cuts, pLs, entropies, Deffs)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "qcut,local_cut,pL,entropy,Deff")
        for i in eachindex(qcuts)
            println(io, "$(qcuts[i]),$(local_cuts[i]),$(pLs[i]),$(entropies[i]),$(Deffs[i])")
        end
    end
    return path
end

function maybe_plot_entropy_curves(path, curves; title="", xlabel="local cut", ylabel="entropy")
    if get(ENV, "CMPS_PLOTS", "0") != "1"
        return nothing
    end

    try
        @eval import Plots

        isempty(curves) && return nothing
        first_curve = curves[1]
        p = Base.invokelatest(
            Plots.plot,
            first_curve.x,
            first_curve.y;
            xlabel,
            ylabel,
            title,
            label=first_curve.label,
            linewidth=2,
        )

        for curve in curves[2:end]
            Base.invokelatest(
                Plots.plot!,
                p,
                curve.x,
                curve.y;
                label=curve.label,
                linewidth=2,
            )
        end

        mkpath(dirname(path))
        Base.invokelatest(Plots.savefig, p, path)
        return path
    catch err
        @warn "CMPS_PLOTS=1 was set, but entropy plotting failed. Inspect CSV output instead." exception=(err, catch_backtrace())
        return nothing
    end
end

function write_phase_csv(path, qgrid, ψ, phase)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "q,density,realpsi,imagpsi,phase")
        for i in eachindex(qgrid)
            println(io, "$(qgrid[i]),$(abs2(ψ[i])),$(real(ψ[i])),$(imag(ψ[i])),$(phase[i])")
        end
    end
    return path
end

function write_momentum_scaling_csv(path, Ps, wavelengths, gridNs, fockProxies, params)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "P,phase_wavelength,grid_N_proxy,fock_proxy,localized_params")
        for i in eachindex(Ps)
            println(io, "$(Ps[i]),$(wavelengths[i]),$(gridNs[i]),$(fockProxies[i]),$(params[i])")
        end
    end
    return path
end

function maybe_plot_wavefunction_phase(path, qgrid, ψ, phase; title="")
    if get(ENV, "CMPS_PLOTS", "0") != "1"
        return nothing
    end

    try
        @eval import Plots

        p = Base.invokelatest(
            Plots.plot,
            qgrid,
            real.(ψ);
            xlabel="q",
            ylabel="Re ψ(q)",
            label="real wavefunction",
            linewidth=2,
            color=:navy,
            title,
            legend=:topright,
        )

        p_phase = Base.invokelatest(Plots.twinx, p)
        Base.invokelatest(
            Plots.plot!,
            p_phase,
            qgrid,
            phase;
            ylabel="unwrapped phase",
            label="phase Pq",
            linewidth=2,
            linestyle=:dash,
            color=:darkorange,
            legend=:bottomright,
        )

        mkpath(dirname(path))
        Base.invokelatest(Plots.savefig, p, path)
        return path
    catch err
        @warn "CMPS_PLOTS=1 was set, but phase plotting failed. Inspect CSV output instead." exception=(err, catch_backtrace())
        return nothing
    end
end

function maybe_plot_wavefunction_phase_comparison(path, cases; title="")
    if get(ENV, "CMPS_PLOTS", "0") != "1"
        return nothing
    end

    try
        @eval import Plots

        isempty(cases) && return nothing

        wave_colors = (:navy, :crimson, :darkgreen, :purple)
        phase_colors = (:darkorange, :seagreen, :brown, :gray40)
        first_case = cases[1]

        p = Base.invokelatest(
            Plots.plot,
            first_case.qgrid,
            real.(first_case.psi);
            xlabel="q",
            ylabel="Re ψ(q)",
            label="Re ψ, $(first_case.label)",
            linewidth=2,
            color=wave_colors[1],
            title,
            legend=:topright,
        )

        for i in 2:length(cases)
            case = cases[i]
            Base.invokelatest(
                Plots.plot!,
                p,
                case.qgrid,
                real.(case.psi);
                label="Re ψ, $(case.label)",
                linewidth=2,
                color=wave_colors[mod1(i, length(wave_colors))],
            )
        end

        p_phase = Base.invokelatest(Plots.twinx, p)
        for (i, case) in enumerate(cases)
            Base.invokelatest(
                Plots.plot!,
                p_phase,
                case.qgrid,
                case.phase;
                ylabel="unwrapped phase",
                label="phase, $(case.label)",
                linewidth=2,
                linestyle=:dash,
                color=phase_colors[mod1(i, length(phase_colors))],
                legend=:bottomright,
            )
        end

        mkpath(dirname(path))
        Base.invokelatest(Plots.savefig, p, path)
        return path
    catch err
        @warn "CMPS_PLOTS=1 was set, but comparison phase plotting failed. Inspect CSV output instead." exception=(err, catch_backtrace())
        return nothing
    end
end
