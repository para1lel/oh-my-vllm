import argparse
import csv
from pathlib import Path


def plot(csv_path, figure_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import PercentFormatter

    with Path(csv_path).open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("no requests to plot")
    background, foreground = "#101720", "#e5edf5"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": foreground,
            "axes.labelcolor": foreground,
            "xtick.color": "#afbdcc",
            "ytick.color": "#afbdcc",
            "axes.edgecolor": "#354353",
            "savefig.facecolor": background,
        }
    )
    figure, axis = plt.subplots(figsize=(13, 7), layout="constrained")
    figure.patch.set_facecolor(background)
    axis.set_facecolor("#141e2a")
    colors = LinearSegmentedColormap.from_list(
        "cache_hit", ["#4085e5", "#46b7cb", "#84d0b0", "#f6de72"]
    )
    points = axis.scatter(
        [float(row["processing_start_s"]) for row in rows],
        [float(row["output_tokens_per_second"]) for row in rows],
        c=[float(row["cache_hit_ratio"]) for row in rows],
        cmap=colors,
        norm=Normalize(0, 1),
        s=35,
        alpha=0.9,
        linewidths=0.35,
        edgecolors="#101720",
        zorder=3,
    )
    axis.set(
        xlabel="Server processing start since profiling began (s)",
        ylabel="Output tokens / server processing time (token/s)",
        xlim=(
            0,
            max(900, max(float(row["processing_start_s"]) for row in rows) * 1.02),
        ),
        ylim=(0, None),
    )
    axis.grid(color="#344153", alpha=0.48, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title(
        "AgentX  |  MTP4  |  One NVIDIA B200",
        loc="left",
        fontsize=20,
        fontweight="bold",
        pad=38,
    )
    axis.text(
        0,
        1.025,
        "4 session trees  /  Up to 4 scheduled requests  /  "
        f"{len(rows):,} completed requests",
        transform=axis.transAxes,
        color="#aebdcd",
        fontsize=11,
    )
    colorbar = figure.colorbar(points, ax=axis, pad=0.025, fraction=0.035)
    colorbar.set_label("Measured prefix cache hit ratio", labelpad=13)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    figure.supxlabel(
        "Processing includes preparation and later scheduling waits. "
        "Warmup excluded; requests may finish during drain.",
        color="#aebdcd",
        fontsize=10,
    )
    Path(figure_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    arguments = parser.parse_args()
    plot(arguments.csv, arguments.figure)


if __name__ == "__main__":
    main()
