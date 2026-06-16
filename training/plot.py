import argparse
import json
import os
from pathlib import Path

from matplotlib import pyplot as plt, lines as mlines

PLOT_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]  # extend if needed

script_dir = Path(os.path.dirname(os.path.abspath(__file__)))


def plot_metric_curves(results, output_dir: Path):
    plt.figure(figsize=(10, 6))

    for i, (model_name, metrics) in enumerate(results.items()):
        if model_name in ["overall", "hyperparameters"]:
            continue  # Skip overall and hyperparameter metrics for plotting
        epochs = range(1, len(metrics["validation_metrics"]) + 1)
        marker = PLOT_MARKERS[i % len(PLOT_MARKERS)]

        plt.plot(
            epochs,
            [
                metrics["validation_metrics"][i]["macro"]["f1"]
                for i in range(len(metrics["validation_metrics"]))
            ],
            color=plt.get_cmap("plasma")(i / len(results)),
            marker=marker,
            mec="white",
            linestyle="-",
        )

        """ plt.plot(
            epochs,
            [metrics["validation_metrics"][i]["micro"]["f1"] for i in range(len(metrics["validation_metrics"]))],
            color="tab:orange",
            marker=marker,
            linestyle="-",
        )

        plt.plot(
            epochs,
            [metrics["validation_metrics"][i]["macro"]["accuracy"] for i in range(len(metrics["validation_metrics"]))],
            color="tab:green",
            marker=marker,
            linestyle="-",
        )

        plt.plot(
            epochs,
            [metrics["validation_metrics"][i]["macro"]["precision"] for i in range(len(metrics["validation_metrics"]))],
            color="tab:red",
            marker=marker,
            linestyle="-",
        )

        plt.plot(
            epochs,
            [metrics["validation_metrics"][i]["macro"]["recall"] for i in range(len(metrics["validation_metrics"]))],
            color="tab:purple",
            marker=marker,
            linestyle="-",
        ) """

    plt.xlabel("Epoch")
    plt.xticks(
        [
            i
            for i in range(
                1,
                max(
                    [len(metrics["validation_metrics"]) for metrics in results.values()]
                )
                + 1,
            )
        ]
    )
    plt.ylabel("Macro F1")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.title("Validation Macro F1 per Epoch")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "validation_metrics.png", dpi=300)


def plot_train_val_loss_curves(results, output_dir: Path):
    train_color = "tab:blue"
    val_color = "tab:orange"

    plt.figure(figsize=(10, 6))

    for i, (model_name, metrics) in enumerate(results.items()):
        if model_name in ["overall", "hyperparameters"]:
            continue  # Skip overall and hyperparameter metrics for plotting
        epochs = range(1, len(metrics["training_loss"]) + 1)
        marker = PLOT_MARKERS[i % len(PLOT_MARKERS)]

        # Train
        plt.plot(
            epochs,
            metrics["training_loss"],
            color=train_color,
            marker=marker,
            mec="white",
            linestyle="-",
            alpha=0.8,
        )

        # Validation
        plt.plot(
            epochs,
            metrics["validation_loss"],
            color=val_color,
            marker=marker,
            mec="white",
            linestyle="--",
            alpha=0.8,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(
        [
            i
            for i in range(
                1,
                max(
                    [len(metrics["validation_metrics"]) for metrics in results.values()]
                )
                + 1,
            )
        ]
    )
    plt.grid(alpha=0.3)
    plt.title("Training and Validation Loss Curves")
    plt.grid(True)

    train_line = mlines.Line2D(
        [], [], color=train_color, linestyle="-", label="Train Loss"
    )
    val_line = mlines.Line2D(
        [], [], color=val_color, linestyle="--", label="Validation Loss"
    )

    plt.legend(handles=[train_line, val_line])
    plt.tight_layout()
    plt.savefig(output_dir / "training_validation_loss_curves.png", dpi=300)


def plot_train_loss_curve(results, output_dir: Path):
    plt.figure(figsize=(8, 5))
    for i, (model_name, metrics) in enumerate(results.items()):
        if model_name in ["overall", "hyperparameters"]:
            continue  # Skip overall and hyperparameter metrics for plotting
        epochs = range(1, len(metrics["training_loss"]) + 1)
        marker = PLOT_MARKERS[i % len(PLOT_MARKERS)]

        plt.plot(
            epochs,
            metrics["training_loss"],
            marker=marker,
            mec="white",
            linestyle="-",
            alpha=0.8,
        )

    plt.legend(
        [
            model_name
            for model_name in results.keys()
            if model_name not in ["overall", "hyperparameters"]
        ],
        loc="upper right",
    )
    plt.xticks(epochs)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_loss_curve.png", dpi=300)
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
        help="Type of plot to generate",
        required=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.results_file, "r") as f:
        results = json.load(f)

    if args.type in ["train_loss", "all"]:
        plot_train_loss_curve(results, Path(args.results_file).parent)
    if args.type in ["val_loss", "all"]:
        plot_train_val_loss_curves(results, Path(args.results_file).parent)
    if args.type in ["metrics", "all"]:
        plot_metric_curves(results, Path(args.results_file).parent)


if __name__ == "__main__":
    main()
