"""Plot/report the GKP figure-pack CSV outputs.

Run after:

    julia --project=. examples/20_gkp_figure_pack.jl

This script intentionally only plots the q-space comb density. The chi sweep
and Hamiltonian-noise outputs are short diagnostic datasets, so they are written
as Markdown tables instead of being dressed up as trend plots.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
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
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.14, top=0.86, hspace=0.18)
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
    fig.suptitle("Finite-energy GKP combs", y=0.965, fontsize=15.5, fontweight="semibold")
    fig.text(
        0.985,
        0.035,
        "FD ground states on q-grid, fast demo data",
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
    chi_path, noise_path = write_tables()
    print(f"wrote {svg_path}")
    print(f"wrote {png_path}")
    print(f"wrote {chi_path}")
    print(f"wrote {noise_path}")


if __name__ == "__main__":
    main()
