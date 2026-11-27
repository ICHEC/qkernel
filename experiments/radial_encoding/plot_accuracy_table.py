"""
Draw a table with the accuracy of the best quantum and classical models.

The numbers are read from `results.json` and `classical_results.json`.

Run from the repository root with:

    uv run python experiments/radial_encoding/plot_accuracy_table.py
"""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from plot_best_run import PLOTS
from run_classical_baseline import CLASSICAL_OUTPUT
from run_experiments import OUTPUT

HEADER_COLOUR = "#14213D"
TEXT_COLOUR = "#1F2937"
MUTED_COLOUR = "#6B7280"
LINE_COLOUR = "#D1D5DB"
ROWS = [
    # Name, description, accent colour, background colour
    ("Quantum kernel SVM", "Radial encoding", "#6D28D9", "#F3EEFC"),
    ("Classical SVM", "RBF kernel", "#0F766E", "#EAF6F4"),
]

WIDTH, HEIGHT = 12.0, 5.2
COLUMNS = {"model": 0.9, "train": 7.2, "test": 9.8}


def main() -> None:
    """Read the selected runs and save the table as an image."""
    quantum = json.loads(OUTPUT.read_text())["selected"]["overall"]
    classical = json.loads(CLASSICAL_OUTPUT.read_text())["selected"]["overall"]
    assert quantum["features"] == classical["features"]
    assert classical["model"] == "svm_rbf"

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")

    def text(x, y, content, size, colour=TEXT_COLOUR, **kwargs):
        ax.text(x, y, content, fontsize=size, color=colour, va="center", **kwargs)

    def box(y, height, colour):
        ax.add_patch(
            FancyBboxPatch(
                (0.4, y),
                WIDTH - 0.8,
                height,
                boxstyle="round,pad=0,rounding_size=0.18",
                facecolor=colour,
                edgecolor="none",
            )
        )

    # Header
    box(3.9, 0.9, HEADER_COLOUR)
    header = {"fontweight": "bold"}
    text(COLUMNS["model"], 4.35, "Model", 24, "white", **header)
    text(COLUMNS["train"], 4.35, "Train", 24, "white", ha="center", **header)
    text(COLUMNS["test"], 4.35, "Test", 24, "white", ha="center", **header)

    # One row per model
    for (name, description, accent, background), run, y in zip(
        ROWS, [quantum, classical], [2.35, 0.95]
    ):
        box(y, 1.25, background)
        ax.add_patch(Rectangle((0.55, y + 0.2), 0.12, 0.85, color=accent))
        text(COLUMNS["model"], y + 0.82, name, 24, fontweight="bold")
        text(COLUMNS["model"], y + 0.38, description, 17, MUTED_COLOUR)
        for column in ["train", "test"]:
            text(
                COLUMNS[column],
                y + 0.62,
                f"{run['results'][f'{column}_accuracy']:.1%}",
                34,
                accent,
                ha="center",
                fontweight="bold",
            )

    # Column separator and footnote
    ax.plot([8.5, 8.5], [1.1, 3.45], color=LINE_COLOUR, linewidth=1.5)
    text(
        WIDTH / 2,
        0.42,
        "Same four bands (B02, B03, B04, B08). "
        "Settings chosen by cross-validation on the training area.",
        14,
        MUTED_COLOUR,
        ha="center",
        style="italic",
    )

    PLOTS.mkdir(exist_ok=True)
    path = PLOTS / "accuracy_table.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print("Saved", path)


if __name__ == "__main__":
    main()
