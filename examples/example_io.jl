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
