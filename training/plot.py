import argparse
import json
import os
from pathlib import Path

from matplotlib import pyplot as plt, lines as mlines
import numpy as np

from utils import console

PLOT_MARKERS = ["s", "^", "D", "v", "P", "X", "p", ">"]  # extend if needed

script_dir = Path(os.path.dirname(os.path.abspath(__file__)))


def plot_validation_metric_curve_single(
    results: dict, label: str, metric: str, output_path: Path
):
    plt.figure(figsize=(10, 6))
    max_epochs = 0
    mean = []
    std = []

    # Plot individual model curves
    for i, (model_name, model_results) in enumerate(results.items()):
        if model_name == "overall":
            continue  # Skip overall for plotting
        if model_results["validation_metrics"][0].get(label) is None:
            return  # Skip if the label is not present in the validation metrics
        if model_results["validation_metrics"][0][label].get(metric) is None:
            return  # Skip if the metric is not present in the validation metrics
        epochs = range(
            1, model_results["best_epoch"] + 1
        )  # +1 to include the best epoch
        marker = PLOT_MARKERS[i % len(PLOT_MARKERS)]
        max_epochs = max(max_epochs, len(epochs))

        plt.plot(
            epochs,
            [
                model_results["validation_metrics"][i][label][metric]
                for i in range(model_results["best_epoch"])
            ],
            color=plt.get_cmap("plasma")(i / (len(results) - 1)),
            marker=marker,
            mec="white",
            linestyle="-",
            label="$M_{%s}$" % model_name[0:4],
            zorder=2,
        )

    results_at_epoch = [[] for _ in range(max_epochs)]

    # Aggregate results across models for each epoch
    for model_name, model_results in results.items():
        if model_name == "overall":
            continue  # Skip overall for plotting
        best_epoch = model_results["best_epoch"]
        for epoch in range(max_epochs):
            if epoch < best_epoch:
                results_at_epoch[epoch].append(
                    model_results["validation_metrics"][epoch][label][metric]
                )
            else:
                results_at_epoch[epoch].append(
                    model_results["validation_metrics"][best_epoch - 1][label][metric]
                )
    for epoch in range(max_epochs):
        mean.append(np.mean(results_at_epoch[epoch]))
        std.append(np.std(results_at_epoch[epoch]))

    mean = np.array(mean)
    std = np.array(std)

    # Plot shaded area for mean ± std
    plt.fill_between(
        range(1, len(mean) + 1),
        mean - std,
        mean + std,
        color="gray",
        alpha=0.3,
        label="Mean ± Std Dev",
        zorder=0,
    )

    # Plot mean line
    plt.plot(
        range(1, len(mean) + 1),
        mean,
        color="black",
        linestyle="--",
        label="Mean",
        zorder=2,
    )

    # Plot the last point of the mean with a marker and annotation
    plt.plot(
        len(mean),
        mean[-1],
        marker="o",
        color="black",
        mec="white",
        zorder=3,
    )
    plt.annotate(
        f"{mean[-1]:.3f}",
        xy=(len(mean), mean[-1]),
        xytext=(len(mean) - 0.01, mean[-1] + 0.02),
        ha="right",
    )

    plt.legend(loc="lower right" if mean[-1] > 0.5 else "upper right")
    plt.xlabel("Epoch")
    plt.xticks(
        [
            i
            for i in range(
                1,
                max_epochs + 1,
            )
        ]
    )
    plt.ylabel(f"{label.capitalize()} {metric.capitalize()}")
    plt.yticks([i / 10 for i in range(11)])
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, zorder=0)
    # plt.title(f"Validation {label.capitalize()}-{metric.capitalize()} per Epoch")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


def plot_validation_metric_curves(results: dict, output_dir: Path):
    console.print("Plotting validation metrics curves...")
    for label in ["macro", "span", "B", "I"]:
        for metric in ["f1", "precision", "recall", "jaccard"]:
            plot_validation_metric_curve_single(
                results,
                label,
                metric,
                output_dir / f"validation_{label}_{metric}.png",
            )


def plot_train_val_loss_curves(results, output_path: Path):
    console.print("Plotting training and validation loss curves...")
    train_color = "tab:blue"
    val_color = "tab:orange"
    max_epochs = 0

    plt.figure(figsize=(10, 6))

    for i, (model_name, metrics) in enumerate(results.items()):
        if model_name == "overall":
            continue  # Skip overall for plotting
        epochs = range(1, metrics["best_epoch"] + 1)  # +1 to include the best epoch
        max_epochs = max(max_epochs, len(epochs))
        marker = PLOT_MARKERS[i % len(PLOT_MARKERS)]

        # Train
        plt.plot(
            epochs,
            metrics["training_loss"][: len(epochs)],
            color=train_color,
            marker=marker,
            mec="white",
            mew=0.7,
            ms=7,
            linestyle="-",
            zorder=2,
        )

        # Validation
        plt.plot(
            epochs,
            metrics["validation_loss"][: len(epochs)],
            color=val_color,
            marker=marker,
            mec="white",
            mew=0.7,
            ms=7,
            linestyle="-",
            zorder=2,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks([i for i in range(1, max_epochs + 1)])
    plt.grid(True, alpha=0.3, zorder=0)
    # plt.title("Training and Validation Loss Curves")

    train_line = mlines.Line2D(
        [], [], color=train_color, linestyle="-", label="Training"
    )
    val_line = mlines.Line2D(
        [], [], color=val_color, linestyle="--", label="Validation"
    )
    marker_lines = [
        mlines.Line2D(
            [],
            [],
            color="black",
            marker=PLOT_MARKERS[i % len(PLOT_MARKERS)],
            linestyle="None",
            label="$M_{%s}$" % k[0:4],
        )
        for i, k in enumerate(results.keys())
        if k != "overall"
    ]

    plt.legend(
        handles=[train_line, val_line] + marker_lines,
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


def plot_train_loss_curve(results, output_path: Path):
    console.print("Plotting training loss curve...")
    plt.figure(figsize=(10, 6))
    for i, (model_name, metrics) in enumerate(results.items()):
        if model_name == "overall":
            continue  # Skip overall and hyperparameter metrics for plotting
        epochs = range(1, metrics["best_epoch"] + 1)  # +1 to include the best epoch
        marker = PLOT_MARKERS[i % len(PLOT_MARKERS)]

        plt.plot(
            epochs,
            metrics["training_loss"][: len(epochs)],
            marker=marker,
            color=plt.get_cmap("plasma")(i / (len(results) - 1)),
            mec="white",
            linestyle="-",
            zorder=2,
        )

    plt.legend(
        [model_name for model_name in results.keys() if model_name != "overall"],
        loc="upper right",
    )
    plt.xticks(epochs)
    # plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot training and validation metrics."
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to the JSON file containing training results",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["train_loss", "val_loss", "metrics", "all"],
        default="all",
        help="Type of plot to generate",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.results_file.endswith("results.json"):
        if Path(args.results_file) / "results.json":
            args.results_file = str(Path(args.results_file) / "results.json")
        else:
            console.print(
                "Error: Results file must be a JSON file named 'results.json'"
            )
            return

    with open(args.results_file, "r") as f:
        results = json.load(f)

    figures_dir = Path(args.results_file).parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    if args.type in ["train_loss", "all"]:
        plot_train_loss_curve(results, figures_dir / "training_loss_curve.png")
    if args.type in ["val_loss", "all"]:
        plot_train_val_loss_curves(
            results, figures_dir / "training_validation_loss_curves.png"
        )
    if args.type in ["metrics", "all"]:
        plot_validation_metric_curves(results, figures_dir)

    console.print(f"Plots saved to {figures_dir}")


if __name__ == "__main__":
    main()
