# Example 18: spatial bipartition entropy for one-body quadrature states.
#
# For a one-particle wavefunction over q, a cut q < qcut | q >= qcut has at most
# two Schmidt weights:
#
#   pL = int_{q < qcut} |psi(q)|^2 dq
#   pR = 1 - pL
#
# so S = -pL log(pL) - pR log(pR) and Deff = exp(S) <= 2. This is not
# inter-qumode entanglement; it is a one-coordinate cut diagnostic. Its value is
# that it separates representation cost from actual cut complexity.
#
# Run:
#   julia --project=. examples/18_spatial_bipartition_entropy.jl
#
# Optional artifacts:
#   CMPS_WRITE_DATA=1 julia --project=. examples/18_spatial_bipartition_entropy.jl
#   CMPS_WRITE_DATA=1 CMPS_PLOTS=1 julia --project=. examples/18_spatial_bipartition_entropy.jl

using Printf
include("example_io.jl")

function binary_entropy(p::Real)
    p = clamp(Float64(p), 0.0, 1.0)
    if p == 0.0 || p == 1.0
        return 0.0
    end
    return -p * log(p) - (1 - p) * log(1 - p)
end

function entropy_from_probability(pL)
    S = binary_entropy(pL)
    return (; pL, entropy=S, Deff=exp(S))
end

erf_float(x) = ccall(:erf, Float64, (Float64,), Float64(x))

gaussian_left_probability(qcut; center=0.0, σ=1.0) =
    0.5 * (1 + erf_float((qcut - center) / σ))

function cat_left_probability(qcut; Q, σ=1.0)
    return 0.5 * gaussian_left_probability(qcut; center=-Q, σ) +
           0.5 * gaussian_left_probability(qcut; center=Q, σ)
end

function gaussian_entropy_curve(Q; σ=1.0, L=6.0, N=601)
    local_cuts = collect(range(-L * σ, L * σ; length=N))
    qcuts = Q .+ local_cuts
    pLs = [gaussian_left_probability(qcut; center=Q, σ) for qcut in qcuts]
    entropies = [binary_entropy(p) for p in pLs]
    Deffs = exp.(entropies)
    return (; qcuts, local_cuts, pLs, entropies, Deffs)
end

function cat_entropy_curve(Q; σ=1.0, Lσ=6.0, N=1201)
    qcuts = collect(range(-Q - Lσ * σ, Q + Lσ * σ; length=N))
    local_cuts = copy(qcuts)
    pLs = [cat_left_probability(qcut; Q, σ) for qcut in qcuts]
    entropies = [binary_entropy(p) for p in pLs]
    Deffs = exp.(entropies)
    return (; qcuts, local_cuts, pLs, entropies, Deffs)
end

function print_row(Q, cut_label, pL)
    d = entropy_from_probability(pL)
    @printf("%11.3e %10s %10.6f %10.6f %10.6f\n",
            Q, cut_label, d.pL, d.entropy, d.Deff)
end

@printf("spatial bipartition entropy for one-body quadrature states\n")
@printf("S(qcut) = -pL log(pL) - (1-pL) log(1-pL), so Deff <= 2\n\n")

@printf("displaced Gaussian center cuts\n")
@printf("%11s %10s %10s %10s %10s\n", "Q", "cut", "pL", "S", "Deff")
for Q in (0.0, 1.0e3, 1.0e6)
    print_row(Q, "Q", gaussian_left_probability(Q; center=Q))
end

Qcat = 20.0
@printf("\ntwo-Gaussian cat cuts\n")
@printf("%11s %10s %10s %10s %10s\n", "Q", "cut", "pL", "S", "Deff")
print_row(Qcat, "-Q", cat_left_probability(-Qcat; Q=Qcat))
print_row(Qcat, "0", cat_left_probability(0.0; Q=Qcat))
print_row(Qcat, "Q", cat_left_probability(Qcat; Q=Qcat))

if get(ENV, "CMPS_WRITE_DATA", "0") == "1"
    g0 = gaussian_entropy_curve(0.0)
    g1e6 = gaussian_entropy_curve(1.0e6)
    cat = cat_entropy_curve(Qcat)

    path = write_entropy_csv(
        joinpath("outputs", "gaussian_bipartition_entropy_Q0.csv"),
        g0.qcuts,
        g0.local_cuts,
        g0.pLs,
        g0.entropies,
        g0.Deffs,
    )
    println("wrote entropy CSV: ", path)

    path = write_entropy_csv(
        joinpath("outputs", "gaussian_bipartition_entropy_Q1e6.csv"),
        g1e6.qcuts,
        g1e6.local_cuts,
        g1e6.pLs,
        g1e6.entropies,
        g1e6.Deffs,
    )
    println("wrote entropy CSV: ", path)

    path = write_entropy_csv(
        joinpath("outputs", "cat_bipartition_entropy_Q20.csv"),
        cat.qcuts,
        cat.local_cuts,
        cat.pLs,
        cat.entropies,
        cat.Deffs,
    )
    println("wrote entropy CSV: ", path)

    plot_path = maybe_plot_entropy_curves(
        joinpath("outputs", "gaussian_bipartition_entropy.svg"),
        [
            (; x=g0.local_cuts, y=g0.entropies, label="Q = 0"),
            (; x=g1e6.local_cuts, y=g1e6.entropies, label="Q = 1e6"),
        ];
        title="Displaced Gaussian cut entropy",
        xlabel="local cut qcut - Q",
        ylabel="S(qcut)",
    )
    if plot_path !== nothing
        println("wrote entropy plot: ", plot_path)
    end

    plot_path = maybe_plot_entropy_curves(
        joinpath("outputs", "cat_bipartition_entropy_Q20.svg"),
        [
            (; x=cat.local_cuts, y=cat.entropies, label="cat Q = 20"),
        ];
        title="Two-Gaussian cat cut entropy plateau",
        xlabel="qcut",
        ylabel="S(qcut)",
    )
    if plot_path !== nothing
        println("wrote entropy plot: ", plot_path)
    end
end
