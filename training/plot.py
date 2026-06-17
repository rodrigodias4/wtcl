import argparse
import json
import os
from pathlib import Path

from matplotlib import pyplot as plt, lines as mlines

PLOT_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]  # extend if needed

script_dir = Path(os.path.dirname(os.path.abspath(__file__)))


def plot_metric_curves(results: dict, label: str, metric: str, output_dir: Path):
    plt.figure(figsize=(10, 6))
    max_epochs = 0

    for i, (model_name, metrics) in enumerate(results.items()):
        if model_name in ["overall", "hyperparameters"]:
            continue  # Skip overall and hyperparameter metrics for plotting
        epochs = range(1, metrics["best_epoch"] + 1)  # +1 to include the best epoch
        marker = PLOT_MARKERS[i % len(PLOT_MARKERS)]
        max_epochs = max(max_epochs, len(epochs))

        plt.plot(
            epochs,
            [
                metrics["validation_metrics"][i][label][metric]
                for i in range(metrics["best_epoch"])
            ],
            color=plt.get_cmap("plasma")(i / len(results)),
            marker=marker,
            mec="white",
            linestyle="-",
        )

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
    plt.grid(True, alpha=0.3)
    plt.title(f"Validation {label.capitalize()}-{metric.capitalize()} per Epoch")

    plt.tight_layout()
    plt.savefig(output_dir / f"validation_{label}_{metric}.png", dpi=300)


def plot_train_val_loss_curves(results, output_dir: Path):
    train_color = "tab:blue"
    val_color = "tab:orange"
    max_epochs = 0

    plt.figure(figsize=(10, 6))

    for i, (model_name, metrics) in enumerate(results.items()):
        if model_name in ["overall", "hyperparameters"]:
            continue  # Skip overall and hyperparameter metrics for plotting
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
            linestyle="-",
            alpha=0.8,
        )

        # Validation
        plt.plot(
            epochs,
            metrics["validation_loss"][: len(epochs)],
            color=val_color,
            marker=marker,
            mec="white",
            linestyle="--",
            alpha=0.8,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks([i for i in range(1, max_epochs + 1)])
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
        epochs = range(1, metrics["best_epoch"] + 1)  # +1 to include the best epoch
        marker = PLOT_MARKERS[i % len(PLOT_MARKERS)]

        plt.plot(
            epochs,
            metrics["training_loss"][: len(epochs)],
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
    
    figures_dir = Path(args.results_file).parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    if args.type in ["train_loss", "all"]:
        plot_train_loss_curve(results, figures_dir)
    if args.type in ["val_loss", "all"]:
        plot_train_val_loss_curves(results, figures_dir)
    if args.type in ["metrics", "all"]:
        for label in ["macro", "B", "I"]:
            for metric in ["f1", "accuracy", "precision", "recall"]:
                plot_metric_curves(
                    results, label, metric, figures_dir
                )


if __name__ == "__main__":
    main()
