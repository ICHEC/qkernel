"""
Plot the true and predicted label grids of the best radial encoding run.

The best run is read from `results.json` and repeated with `qkernel.Pipeline`.
Train predictions are made on the data the model was trained on.

Run from the repository root with:

    uv run python experiments/radial_encoding/plot_best_run.py
"""

import json
import tempfile
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from run_experiments import OUTPUT, load_datasets, make_pipeline

from qkernel.model import SVM
from qkernel.utils import plot_label_grid_with_points

PLOTS = OUTPUT.parent / "plots"

# Colours in the order of the numeric labels
CLASS_COLOURS = [
    "#E53935",  # 1: urban (red)
    "#F0C419",  # 2: agricultural (gold)
    "#228B22",  # 3: forest (green)
]
COLORMAP = "land_cover"
matplotlib.colormaps.register(ListedColormap(CLASS_COLOURS, name=COLORMAP))


def make_grid(df, labels):
    """Dataframe with `labels` and the grid ranks, sorted as the plot expects."""
    return df.assign(
        plot_label=labels,
        lat_rank=df["latitude"].rank(method="dense").astype(int) - 1,
        lon_rank=df["longitude"].rank(method="dense").astype(int) - 1,
    ).sort_values(by=["lat_rank", "lon_rank"])


def class_names(df) -> list:
    """Class names in the order of their numeric labels."""
    classes = df[["label", "class"]].drop_duplicates().sort_values("label")
    return list(classes["class"])


def save_label_grid(df, labels, title: str, filename: str) -> None:
    """Plot `labels` on the spatial grid of `df` and save the figure."""

    def save() -> None:
        plt.title(title)
        # Keep the legend outside of the grid
        plt.gca().get_legend().set_bbox_to_anchor((1.02, 1))
        plt.savefig(PLOTS / filename, dpi=150, bbox_inches="tight")
        plt.close()

    # The plotting function ends with `plt.show`, which we replace to save instead
    show, plt.show = plt.show, save
    try:
        plot_label_grid_with_points(
            make_grid(df, labels),
            label_col="plot_label",
            label_names=class_names(df),
            point_size=6,
            cmap_name=COLORMAP,
        )
    finally:
        plt.show = show


def save_comparison(df, predicted, title: str, filename: str) -> None:
    """Save the true and predicted label grids side by side in one figure."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 11.5), layout="constrained")
    for ax, labels, panel_title in zip(
        axes, [df["label"], predicted], ["True labels", "Predicted labels"]
    ):
        plot_label_grid_with_points(
            make_grid(df, labels),
            label_col="plot_label",
            label_names=class_names(df),
            point_size=6,
            cmap_name=COLORMAP,
            ax=ax,
        )
        ax.set_title(panel_title, fontsize=22)
        ax.set_xlabel("Longitude Rank", fontsize=20)
        ax.set_ylabel("Latitude Rank", fontsize=20)
        ax.tick_params(labelsize=15)

    # Both panels have the same legend, so we show it only once below them
    handles = axes[0].get_legend().legend_handles
    for ax in axes:
        ax.get_legend().remove()
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncols=len(handles),
        fontsize=20,
    )
    fig.suptitle(title, fontsize=28)
    fig.savefig(PLOTS / filename, dpi=150)
    plt.close(fig)


def main() -> None:
    """Repeat the best run and save the label grids."""
    warnings.filterwarnings("ignore", message="divide by zero")
    run = json.loads(OUTPUT.read_text())["selected"]["overall"]
    data = load_datasets()

    with tempfile.TemporaryDirectory() as path:
        run["args"]["results"] = {"path": path, "name": "best_run"}
        pipeline = make_pipeline(run, data)
        pipeline.run()

    # The pipeline only keeps the test predictions
    model = SVM(**run["args"]["model"]["kwargs"])
    model.train(pipeline.gram_train, pipeline.y_train)
    predicted_train = model.predict(pipeline.gram_train)

    assert pipeline.predicted_labels.tolist() == run["results"]["predicted_labels"]
    assert (predicted_train == pipeline.y_train).mean() == pipeline.train_accuracy

    PLOTS.mkdir(exist_ok=True)
    save_comparison(
        data["train"],
        predicted_train,
        "Train dataset",
        "train_true_vs_predicted_labels.png",
    )
    for name, predicted, accuracy in [
        ("train", predicted_train, pipeline.train_accuracy),
        ("test", pipeline.predicted_labels, pipeline.test_accuracy),
    ]:
        df = data[name]
        save_label_grid(
            df, df["label"], f"True {name} labels", f"{name}_true_labels.png"
        )
        save_label_grid(
            df,
            predicted,
            f"Predicted {name} labels (accuracy {accuracy:.3f})",
            f"{name}_predicted_labels.png",
        )
        print(name, "accuracy", accuracy)
    print("Saved plots to", PLOTS)


if __name__ == "__main__":
    main()
