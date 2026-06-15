import argparse
import os
from pathlib import Path

from matplotlib import pyplot as plt

PLOT_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]  # extend if needed
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))


def plot_metric_curves(results, output_file: Path):
    plt.figure(figsize=(10, 6))

    for i, (model_name, metrics) in enumerate(results.items()):
        if model_name == "overall":
            continue  # Skip overall metrics for plotting
        epochs = range(1, len(metrics["validation_metrics"]) + 1)
        marker = PLOT_MARKERS[i % len(PLOT_MARKERS)]

        plt.plot(
            epochs,
            metrics["validation_metrics"]["macro"]["f1"],
            color="tab:blue",
            marker=marker,
            linestyle="-",
        )

        plt.plot(
            epochs,
            metrics["validation_metrics"]["micro"]["f1"],
            color="tab:orange",
            marker=marker,
            linestyle="-",
        )

        plt.plot(
            epochs,
            metrics["validation_metrics"]["macro"]["accuracy"],
            color="tab:green",
            marker=marker,
            linestyle="-",
        )

        plt.plot(
            epochs,
            metrics["validation_metrics"]["macro"]["precision"],
            color="tab:red",
            marker=marker,
            linestyle="-",
        )

        plt.plot(
            epochs,
            metrics["validation_metrics"]["macro"]["recall"],
            color="tab:purple",
            marker=marker,
            linestyle="-",
        )

    plt.xlabel("Epoch")
    plt.xticks(epochs)
    plt.ylabel("Macro F1")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.title("Validation Macro F1 per Epoch")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)


def plot_train_val_loss_curves(results, output_file: Path):
    train_color = "tab:blue"
    val_color = "tab:orange"

    plt.figure(figsize=(10, 6))

    for i, (model_name, metrics) in enumerate(results.items()):
        if model_name == "overall":
            continue  # Skip overall metrics for plotting
        epochs = range(1, len(metrics["train_loss"]) + 1)
        marker = PLOT_MARKERS[i % len(PLOT_MARKERS)]

        # Train
        plt.plot(
            epochs,
            metrics["train_loss"],
            color=train_color,
            marker=marker,
            linestyle="-",
            alpha=0.8,
        )

        # Validation
        plt.plot(
            epochs,
            metrics["validation_loss"],
            color=val_color,
            marker=marker,
            linestyle="--",
            alpha=0.8,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    plt.grid(alpha=0.3)
    plt.title("Training and Validation Loss Curves")
    plt.grid(True)

    train_line = plt.mlines.Line2D(
        [], [], color=train_color, linestyle="-", label="Train Loss"
    )
    val_line = plt.mlines.Line2D(
        [], [], color=val_color, linestyle="--", label="Validation Loss"
    )

    plt.legend(handles=[train_line, val_line])
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)


def plot_train_loss_curve(results, output_file: Path):
    plt.figure(figsize=(8, 5))
    for i, (model_name, metrics) in enumerate(results.items()):
        if model_name == "overall":
            continue  # Skip overall metrics for plotting
        epochs = range(1, len(metrics["training_loss"]) + 1)
        marker = PLOT_MARKERS[i % len(PLOT_MARKERS)]

        plt.plot(
            epochs,
            metrics["training_loss"],
            color="tab:blue",
            marker=marker,
            linestyle="-",
            alpha=0.8,
        )

    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_confusion_matrix(results: dict, output_file: Path):
    # Placeholder for confusion matrix plotting logic
    pass

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
        choices=["train_loss", "val_loss", "metrics", "all"],
        help="Type of plot to generate",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=script_dir / "figures",
        help="Directory to save the generated plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.type in ["train_loss", "all"]:
        plot_train_loss_curve(
            args.results_file, Path(args.output_dir) / "training_loss_curve.png"
        )
    if args.type in ["val_loss", "all"]:
        plot_train_val_loss_curves(
            args.results_file, Path(args.output_dir) / "validation_loss_curve.png"
        )
    if args.type in ["metrics", "all"]:
        plot_metric_curves(
            args.results_file, Path(args.output_dir) / "validation_metrics.png"
        )


if __name__ == "__main__":
    main()
