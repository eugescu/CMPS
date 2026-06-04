# Example 5: constant-gauge χ > 1 variational family.
# This is the first controlled matrix-valued ansatz:
#
#   ψ(q) = exp(-γq²) vL' exp(A(q-qmin)) B exp(A(qmax-q)) vR
#   A = -0.5*C'C - im*K, K = K'
#
# The harmonic oscillator is intentionally the first target. Extra matrix
# capacity should not destabilize the simple Gaussian limit.
#
# Run:
#   julia --project=. examples/05_constant_gauge_harmonic.jl

using Optim
using Printf
using Random
include("../src/ContinuumQuadratureCMPS.jl")
using .ContinuumQuadratureCMPS

Random.seed!(1)

qmin, qmax = -8.0, 8.0
grid_spec = GridSpec(qmin, qmax, 401)
Hdesc = harmonic_hamiltonian()
H, qgrid = finite_difference_hamiltonian(Hdesc, grid_spec)

fam = ConstantGaugeCMPSFamily(2, qmin, qmax)
θ0 = random_constant_gauge_theta(fam; scale=0.05, gamma=0.5)

loss(θ) = constant_gauge_energy(θ, fam, qgrid, H)

result = optimize(loss, θ0, NelderMead(),
                  Optim.Options(iterations=700, show_trace=false))

θopt = Optim.minimizer(result)
ψ = normalized_amplitudes_constant_gauge(qgrid, θopt, fam)
baseline = grid_eigenstates(Hdesc, grid_spec; nev=1)
Eopt = loss(θopt)

@printf("constant-gauge harmonic benchmark\n")
@printf("χ                         = %d\n", fam.χ)
@printf("initial energy            = %.12f\n", loss(θ0))
@printf("optimized energy          = %.12f\n", Eopt)
@printf("finite-difference E0      = %.12f\n", baseline.energies[1])
@printf("expected continuum E0     = %.12f\n", 0.5)
@printf("normalization check       = %.12f\n", real(grid_inner(qgrid, ψ, ψ)))
@printf("residual norm             = %.12e\n", grid_residual_norm(qgrid, H, ψ, Eopt))
