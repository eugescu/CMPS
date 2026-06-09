"""Plot/report the GKP figure-pack CSV outputs.

Run after:

    julia --project=. examples/20_gkp_figure_pack.jl

This script plots doublet-aware q-space comb density, q/p code-sector density,
and a Husimi Q phase-space heatmap. The chi sweep and Hamiltonian-noise outputs
are short diagnostic datasets, so they are written as Markdown tables instead of
being dressed up as trend plots.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def read_rows(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="") as f:
        return list(csv.DictReader(f))


def write_gkp_density_plot() -> tuple[Path, Path]:
    grouped: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for row in read_rows("gkp_density_curves.csv"):
        grouped[float(row["kappa"])].append((float(row["q"]), float(row["density"])))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "svg.fonttype": "none",
        }
    )

    kappas = sorted(grouped)
    fig, axes = plt.subplots(len(kappas), 1, figsize=(8.2, 4.9), sharex=True)
    fig.subplots_adjust(left=0.095, right=0.965, bottom=0.14, top=0.86, hspace=0.18)
    if len(kappas) == 1:
        axes = [axes]

    colors = ["#0f766e", "#b45309", "#2563eb"]
    for ax, kappa, color in zip(axes, kappas, colors):
        data = sorted(grouped[kappa])
        q = [x for x, _ in data]
        density = [y for _, y in data]
        peak = max(density)
        scaled = [d / peak for d in density]

        ax.fill_between(q, scaled, color=color, alpha=0.18, linewidth=0)
        ax.plot(q, scaled, color=color, lw=2.0)
        ax.text(
            0.02,
            0.80,
            rf"$\kappa={kappa:g}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=color,
            fontsize=12,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
        )
        ax.set_ylim(-0.035, 1.08)
        ax.set_yticks([0, 0.5, 1.0])
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.75)
        ax.grid(axis="y", color="#f0eee8", linewidth=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("normalized\ndensity")

    axes[-1].set_xlabel(r"position $q$")
    axes[-1].set_xlim(-10, 10)
    axes[-1].xaxis.set_major_locator(MaxNLocator(9))
    fig.suptitle("Finite-energy GKP code-sector combs", y=0.965,
                 fontsize=15.5, fontweight="semibold")
    fig.text(
        0.985,
        0.035,
        r"plotted density: $\rho_{\rm code}(q)=\frac{1}{2}(|\phi_1|^2+|\phi_2|^2)$",
        ha="right",
        va="bottom",
        color="#6b7280",
        fontsize=8.5,
    )

    OUT.mkdir(exist_ok=True)
    svg_path = OUT / "gkp_density_matplotlib.svg"
    png_path = OUT / "gkp_density_matplotlib.png"
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=240)
    return svg_path, png_path


def add_sqrt_pi_guides(ax) -> None:
    qmax = 10.0
    step = sqrt(3.141592653589793)
    nmax = int(qmax / step)
    for n in range(-nmax, nmax + 1):
        ax.axvline(n * step, color="#d1d5db", lw=0.65, zorder=0)


def write_gkp_doublet_plot() -> tuple[Path, Path]:
    grouped: dict[float, list[dict[str, float]]] = defaultdict(list)
    for row in read_rows("gkp_density_curves.csv"):
        grouped[float(row["kappa"])].append(
            {
                "q": float(row["q"]),
                "rho1": float(row["rho1"]),
                "rho2": float(row["rho2"]),
                "density": float(row["density"]),
            }
        )

    kappa = min(grouped)
    data = sorted(grouped[kappa], key=lambda r: r["q"])
    q = [row["q"] for row in data]
    rho1 = [row["rho1"] for row in data]
    rho2 = [row["rho2"] for row in data]
    code = [row["density"] for row in data]

    branch_peak = max(max(rho1), max(rho2))
    code_peak = max(code)
    rho1 = [x / branch_peak for x in rho1]
    rho2 = [x / branch_peak for x in rho2]
    code = [x / code_peak for x in code]

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.1), sharex=True)
    fig.subplots_adjust(left=0.095, right=0.965, bottom=0.13, top=0.86, hspace=0.24)

    ax = axes[0]
    add_sqrt_pi_guides(ax)
    ax.plot(q, rho1, color="#0f766e", lw=2.0, label=r"$|\phi_1(q)|^2$")
    ax.plot(q, rho2, color="#b45309", lw=2.0, label=r"$|\phi_2(q)|^2$")
    ax.fill_between(q, rho1, color="#0f766e", alpha=0.10, linewidth=0)
    ax.fill_between(q, rho2, color="#b45309", alpha=0.10, linewidth=0)
    ax.set_title(rf"Doublet branches at $\kappa={kappa:g}$", loc="left", fontsize=12.5)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    add_sqrt_pi_guides(ax)
    ax.fill_between(q, code, color="#2563eb", alpha=0.18, linewidth=0)
    ax.plot(q, code, color="#2563eb", lw=2.0)
    ax.set_title(r"Code-sector density $\rho_{\rm code}(q)=\frac{1}{2}(|\phi_1|^2+|\phi_2|^2)$",
                 loc="left", fontsize=12.5)

    for ax in axes:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-0.035, 1.08)
        ax.set_yticks([0, 0.5, 1.0])
        ax.grid(axis="y", color="#f0eee8", linewidth=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("normalized density")

    axes[-1].set_xlabel(r"position $q$")
    axes[-1].xaxis.set_major_locator(MaxNLocator(9))
    fig.suptitle("Finite-energy GKP doublet", y=0.965, fontsize=15.5, fontweight="semibold")
    fig.text(0.985, 0.035, r"vertical guides: $q=n\sqrt{\pi}$",
             ha="right", va="bottom", color="#6b7280", fontsize=8.5)

    svg_path = OUT / "gkp_doublet_code_density_matplotlib.svg"
    png_path = OUT / "gkp_doublet_code_density_matplotlib.png"
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=240)
    return svg_path, png_path


def write_gkp_qp_code_plot() -> tuple[Path, Path]:
    grouped: dict[float, list[dict[str, float]]] = defaultdict(list)
    for row in read_rows("gkp_density_curves.csv"):
        grouped[float(row["kappa"])].append(
            {
                "q": float(row["q"]),
                "density": float(row["density"]),
                "realphi1": float(row["realphi1"]),
                "imagphi1": float(row["imagphi1"]),
                "realphi2": float(row["realphi2"]),
                "imagphi2": float(row["imagphi2"]),
            }
        )

    kappa = min(grouped)
    data = sorted(grouped[kappa], key=lambda r: r["q"])
    q = np.array([row["q"] for row in data])
    rho_q = np.array([row["density"] for row in data])
    phi1 = np.array([row["realphi1"] + 1j * row["imagphi1"] for row in data])
    phi2 = np.array([row["realphi2"] + 1j * row["imagphi2"] for row in data])

    dq = q[1] - q[0]
    p = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(q), d=dq))

    def fft_density(psi: np.ndarray) -> np.ndarray:
        psi_p = dq / np.sqrt(2 * np.pi) * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi)))
        return np.abs(psi_p) ** 2

    rho_p = 0.5 * (fft_density(phi1) + fft_density(phi2))
    rho_q = rho_q / np.max(rho_q)
    rho_p = rho_p / np.max(rho_p)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.1), sharex=False)
    fig.subplots_adjust(left=0.095, right=0.965, bottom=0.13, top=0.86, hspace=0.32)

    ax = axes[0]
    add_sqrt_pi_guides(ax)
    ax.fill_between(q, rho_q, color="#0f766e", alpha=0.18, linewidth=0)
    ax.plot(q, rho_q, color="#0f766e", lw=2.0)
    ax.set_xlim(-10, 10)
    ax.set_title(r"code-sector density in $q$", loc="left", fontsize=12.5)

    ax = axes[1]
    ax.fill_between(p, rho_p, color="#7c3aed", alpha=0.18, linewidth=0)
    ax.plot(p, rho_p, color="#7c3aed", lw=2.0)
    ax.set_xlim(-12, 12)
    ax.set_xlabel(r"momentum $p$")
    ax.set_title(r"FFT density in $p$", loc="left", fontsize=12.5)

    for ax in axes:
        ax.set_ylim(-0.035, 1.08)
        ax.set_yticks([0, 0.5, 1.0])
        ax.grid(color="#f0eee8", linewidth=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("normalized density")

    fig.suptitle(rf"GKP code-sector combs in $q$ and $p$ ($\kappa={kappa:g}$)",
                 y=0.965, fontsize=15.5, fontweight="semibold")
    fig.text(0.985, 0.035, r"$p$ density from FFT of the two FD doublet states",
             ha="right", va="bottom", color="#6b7280", fontsize=8.5)

    svg_path = OUT / "gkp_qp_code_density_matplotlib.svg"
    png_path = OUT / "gkp_qp_code_density_matplotlib.png"
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=240)
    return svg_path, png_path


def trapz_weights(x: np.ndarray) -> np.ndarray:
    dx = float(x[1] - x[0])
    weights = np.full(len(x), dx)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return weights


def coherent_state_on_grid(x: np.ndarray, q0: float, p0: float) -> np.ndarray:
    phase = np.exp(1j * p0 * (x - 0.5 * q0))
    envelope = np.exp(-0.5 * (x - q0) ** 2)
    return np.pi ** (-0.25) * envelope * phase


def check_husimi_grid(q_func: np.ndarray, q_centers: np.ndarray, p_centers: np.ndarray) -> float:
    q_min = float(np.min(q_func))
    q_max = float(np.max(q_func))
    if q_min < -1e-12:
        raise ValueError(f"Husimi grid has negative density below tolerance: {q_min}")
    if not np.all(np.isfinite(q_func)):
        raise ValueError("Husimi grid contains non-finite values")
    if q_max <= 0:
        raise ValueError("Husimi grid has zero peak density")

    mass_qp = float(np.trapezoid(np.trapezoid(q_func, q_centers, axis=1), p_centers))
    if not np.isfinite(mass_qp) or mass_qp <= 0:
        raise ValueError(f"Husimi grid has invalid phase-space mass: {mass_qp}")
    return mass_qp


def write_gkp_husimi_plot() -> tuple[Path, Path]:
    grouped: dict[float, list[dict[str, float]]] = defaultdict(list)
    for row in read_rows("gkp_density_curves.csv"):
        grouped[float(row["kappa"])].append(
            {
                "q": float(row["q"]),
                "realphi1": float(row["realphi1"]),
                "imagphi1": float(row["imagphi1"]),
                "realphi2": float(row["realphi2"]),
                "imagphi2": float(row["imagphi2"]),
            }
        )

    kappa = min(grouped)
    data = sorted(grouped[kappa], key=lambda r: r["q"])
    x = np.array([row["q"] for row in data])
    phi1 = np.array([row["realphi1"] + 1j * row["imagphi1"] for row in data])
    phi2 = np.array([row["realphi2"] + 1j * row["imagphi2"] for row in data])
    weights = trapz_weights(x)

    q_centers = np.linspace(-6.5, 6.5, 121)
    p_centers = np.linspace(-6.5, 6.5, 121)
    q_func = np.empty((len(p_centers), len(q_centers)))

    for ip, p0 in enumerate(p_centers):
        for iq, q0 in enumerate(q_centers):
            coh = coherent_state_on_grid(x, q0, p0)
            overlap1 = np.sum(weights * np.conjugate(coh) * phi1)
            overlap2 = np.sum(weights * np.conjugate(coh) * phi2)
            q_func[ip, iq] = 0.5 * (abs(overlap1) ** 2 + abs(overlap2) ** 2) / np.pi

    mass_qp = check_husimi_grid(q_func, q_centers, p_centers)
    q_func /= np.max(q_func)

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    fig.subplots_adjust(left=0.12, right=0.86, bottom=0.12, top=0.88)
    image = ax.imshow(
        q_func,
        origin="lower",
        extent=[q_centers[0], q_centers[-1], p_centers[0], p_centers[-1]],
        cmap="magma",
        norm=PowerNorm(gamma=0.55, vmin=0.0, vmax=1.0),
        interpolation="bilinear",
        aspect="equal",
    )
    ax.contour(q_centers, p_centers, q_func, levels=[0.2, 0.4, 0.6, 0.8],
               colors="white", linewidths=0.55, alpha=0.34)
    ax.set_title(rf"Husimi $Q$ phase-space density ($\kappa={kappa:g}$)",
                 fontsize=14.5, fontweight="semibold")
    ax.set_xlabel(r"coherent-state center $q$")
    ax.set_ylabel(r"coherent-state center $p$")
    ax.grid(color="white", alpha=0.12, linewidth=0.6)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035,
                        ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    cbar.set_label("normalized Q")
    fig.text(
        0.98,
        0.035,
        rf"code-sector average over the two FD doublet states; q,p-window mass = {mass_qp:.3g}",
        ha="right",
        va="bottom",
        color="#6b7280",
        fontsize=8.5,
    )

    svg_path = OUT / "gkp_husimi_Q_kappa005.svg"
    png_path = OUT / "gkp_husimi_Q_kappa005.png"
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=240)
    return svg_path, png_path


def markdown_table(rows: list[dict[str, str]], columns: list[tuple[str, str, str]]) -> str:
    header = "| " + " | ".join(label for label, _, _ in columns) + " |"
    sep = "| " + " | ".join("---:" for _ in columns) + " |"
    body = []
    for row in rows:
        vals = []
        for _, key, fmt in columns:
            vals.append(format(float(row[key]), fmt))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def write_tables() -> tuple[Path, Path]:
    chi_table = markdown_table(
        read_rows("gkp_accuracy_vs_chi.csv"),
        [
            ("kappa", "kappa", ".2f"),
            ("chi", "chi", ".0f"),
            ("E error", "Eerr", ".4f"),
            ("F2", "F2", ".4f"),
            ("residual", "residual", ".4f"),
        ],
    )
    noise_table = markdown_table(
        read_rows("gkp_noise_response.csv"),
        [
            ("eta", "eta", ".4g"),
            ("E error", "Eerr", ".4f"),
            ("Feta2", "Feta2", ".4f"),
            ("Fclean2", "Fclean2", ".4f"),
            ("residual", "residual", ".4f"),
        ],
    )

    chi_path = OUT / "gkp_chi_table.md"
    noise_path = OUT / "gkp_noise_table.md"
    chi_path.write_text(chi_table, encoding="utf-8")
    noise_path.write_text(noise_table, encoding="utf-8")
    return chi_path, noise_path


def main() -> None:
    svg_path, png_path = write_gkp_density_plot()
    doublet_svg_path, doublet_png_path = write_gkp_doublet_plot()
    qp_svg_path, qp_png_path = write_gkp_qp_code_plot()
    husimi_svg_path, husimi_png_path = write_gkp_husimi_plot()
    chi_path, noise_path = write_tables()
    print(f"wrote {svg_path}")
    print(f"wrote {png_path}")
    print(f"wrote {doublet_svg_path}")
    print(f"wrote {doublet_png_path}")
    print(f"wrote {qp_svg_path}")
    print(f"wrote {qp_png_path}")
    print(f"wrote {husimi_svg_path}")
    print(f"wrote {husimi_png_path}")
    print(f"wrote {chi_path}")
    print(f"wrote {noise_path}")


if __name__ == "__main__":
    main()
