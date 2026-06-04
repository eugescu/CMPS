# Example 10: one-mode displacement errors on the regularized GKP code sector.
#
# This is the first CV error-channel diagnostic built from the gate layer. It
# applies small X(q-shift) and Z(p-shift) displacements to the clean FD
# regularized-GKP ground state, then measures retention in the clean low-energy
# doublet and stabilizer/width diagnostics.
#
# Run:
#   julia --project=. examples/10_one_mode_gkp_errors.jl

using Printf
include("../src/ContinuumQuadratureCMPS.jl")
using .ContinuumQuadratureCMPS

qmin, qmax = -10.0, 10.0
ε = 1.0
α = 2sqrt(pi)
κ = 0.05
shifts = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]

grid_spec = GridSpec(qmin, qmax, 401)
qgrid = grid(grid_spec)
Hdesc = regularized_gkp_hamiltonian(; ε, κ, α, boundary=:zero)
fd = grid_eigenstates(Hdesc, grid_spec; nev=4)
ψ0 = copy(fd.wavefunctions[:, 1])
normalize_grid_state!(qgrid, ψ0)

function report_error_sweep(name, gate_builder)
    println()
    println(name)
    @printf("%9s %10s %10s %10s %10s %10s %10s\n",
            "shift", "Fclean2", "<cosq>", "<cosp>", "<q²>", "<p²>", "bdy")

    for s in shifts
        ψ = apply_gate_to_grid(gate_builder(s), qgrid, ψ0)
        Fclean2 = grid_subspace_overlap_abs2(qgrid, ψ, fd.wavefunctions; nstates=2)
        d = gkp_diagnostics(qgrid, ψ; α, boundary=:zero)
        @printf("%9.3f %10.6f %10.6f %10.6f %10.4f %10.4f %10.3e\n",
                s, Fclean2, d.cosq, d.cosp, d.q2, d.p2, d.boundary_weight)
    end
end

@printf("one-mode displacement errors on regularized GKP FD state\n")
@printf("ε                         = %.6f\n", ε)
@printf("α                         = %.12f\n", α)
@printf("κ                         = %.6f\n", κ)
@printf("FD E0                     = %.12f\n", fd.energies[1])
@printf("FD doublet gap            = %.6e\n", fd.energies[2] - fd.energies[1])
@printf("FD next-sector gap        = %.6e\n", fd.energies[3] - fd.energies[1])

report_error_sweep("X displacement: ψ(q) -> ψ(q - s)", s -> XDisplacementGate(s))
report_error_sweep("Z displacement: ψ(q) -> exp(i s q) ψ(q)", s -> ZDisplacementGate(s))
