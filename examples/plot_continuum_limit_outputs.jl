using DelimitedFiles
using Plots

function read_csv_with_header(path)
    raw = readdlm(path, ',', String)
    header = vec(raw[1, :])
    data = raw[2:end, :]

    cols = Dict{String, Vector{Float64}}()
    for (j, name) in enumerate(header)
        cols[name] = parse.(Float64, data[:, j])
    end

    return cols
end

function ensure_outputs()
    mkpath("outputs")
end

function plot_density_csv(input_path, output_path; title="")
    cols = read_csv_with_header(input_path)

    x = cols["x"]
    ρ = cols["density"]

    p = plot(
        x,
        ρ;
        xlabel = "local coordinate",
        ylabel = "|ψ(q)|²",
        title = title,
        legend = false,
        linewidth = 2,
    )

    savefig(p, output_path)
    println("wrote ", output_path)
end

function gaussian_wavefunction(local_x)
    return π^(-1 / 4) .* exp.(-0.5 .* (local_x .^ 2))
end

function plot_gaussian_wavefunction_entropy(output_path)
    g0 = read_csv_with_header("outputs/gaussian_bipartition_entropy_Q0.csv")
    g1e6 = read_csv_with_header("outputs/gaussian_bipartition_entropy_Q1e6.csv")

    x = g0["local_cut"]
    ψ = gaussian_wavefunction(x)

    p = plot(
        x,
        ψ;
        xlabel = "local coordinate q - Q",
        ylabel = "ψ(q)",
        label = "Gaussian wavefunction",
        linewidth = 2,
        color = :navy,
        title = "Displaced Gaussian: wavefunction and cut entropy",
        legend = :topright,
    )

    p_entropy = twinx(p)

    plot!(
        p_entropy,
        x,
        g0["entropy"];
        ylabel = "S(qcut)",
        label = "entropy Q = 0",
        linewidth = 2,
        linestyle = :dash,
        color = :darkorange,
        legend = :bottomright,
    )

    plot!(
        p_entropy,
        g1e6["local_cut"],
        g1e6["entropy"];
        label = "entropy Q = 1e6",
        linewidth = 2,
        linestyle = :dot,
        color = :crimson,
        legend = :bottomright,
    )

    savefig(p, output_path)
    println("wrote ", output_path)
end

function plot_cat_wavefunction_entropy(output_path)
    density = read_csv_with_header("outputs/two_gaussian_cat_Q20_density.csv")
    entropy = read_csv_with_header("outputs/cat_bipartition_entropy_Q20.csv")

    p = plot(
        density["q"],
        density["realpsi"];
        xlabel = "q",
        ylabel = "ψ(q)",
        label = "cat wavefunction",
        linewidth = 2,
        color = :navy,
        title = "Two separated Gaussians: wavefunction and cut entropy",
        legend = :topright,
    )

    p_entropy = twinx(p)

    plot!(
        p_entropy,
        entropy["qcut"],
        entropy["entropy"];
        ylabel = "S(qcut)",
        label = "cut entropy",
        linewidth = 2,
        linestyle = :dash,
        color = :darkorange,
        legend = :bottomright,
    )

    savefig(p, output_path)
    println("wrote ", output_path)
end

function plot_cat_scaling(input_path, output_path)
    cols = read_csv_with_header(input_path)

    Q = cols["Q"]
    gridN = cols["uniform_grid_N"]
    fock = cols["fock_proxy"]
    params = cols["localized_params"]

    p = plot(
        Q,
        gridN;
        xscale = :log10,
        yscale = :log10,
        xlabel = "packet center Q",
        ylabel = "size / proxy",
        label = "uniform grid points",
        linewidth = 2,
        marker = :circle,
        title = "Two-Gaussian cat representation scaling",
    )

    plot!(
        p,
        Q,
        fock;
        label = "Fock cutoff proxy",
        linewidth = 2,
        marker = :square,
    )

    plot!(
        p,
        Q,
        params;
        label = "localized continuum parameters",
        linewidth = 2,
        marker = :diamond,
    )

    savefig(p, output_path)
    println("wrote ", output_path)
end

function plot_momentum_wavefunction_phase(input_path, output_path)
    cols = read_csv_with_header(input_path)

    p = plot(
        cols["q"],
        cols["realpsi"];
        xlabel = "q",
        ylabel = "Re ψ(q)",
        label = "real wavefunction",
        linewidth = 2,
        color = :navy,
        title = "Momentum displacement: wavefunction and phase",
        legend = :topright,
    )

    p_phase = twinx(p)

    plot!(
        p_phase,
        cols["q"],
        cols["phase"];
        ylabel = "unwrapped phase",
        label = "phase Pq",
        linewidth = 2,
        linestyle = :dash,
        color = :darkorange,
        legend = :bottomright,
    )

    savefig(p, output_path)
    println("wrote ", output_path)
end

ensure_outputs()

plot_gaussian_wavefunction_entropy(
    "outputs/gaussian_wavefunction_entropy_overlay.svg",
)

plot_cat_wavefunction_entropy(
    "outputs/two_gaussian_cat_wavefunction_entropy_overlay.svg",
)

plot_density_csv(
    "outputs/two_gaussian_cat_Q20_density.csv",
    "outputs/two_gaussian_cat_Q20_density.svg";
    title = "Two separated Gaussian packets",
)

plot_cat_scaling(
    "outputs/two_gaussian_cat_scaling.csv",
    "outputs/two_gaussian_cat_scaling.svg",
)

if isfile("outputs/momentum_displacement_P50_phase.csv")
    plot_momentum_wavefunction_phase(
        "outputs/momentum_displacement_P50_phase.csv",
        "outputs/momentum_displacement_P50_real_phase.svg",
    )
end
