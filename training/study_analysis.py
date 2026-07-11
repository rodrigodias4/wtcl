from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from json import load
import colormaps as cmaps
from utils import console


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "input_file", type=str, help="Path to the input file containing study data."
    )

    return parser.parse_args()


def plot_result_per_trial(study_results, outdir):
    # Extract the trial numbers and corresponding objective values

    trial_numbers = []
    objective_values = []
    num_completed_folds = []
    for trial_number, trial_results in study_results["trials"].items():
        trial_numbers.append(trial_number)
        objective_values.append(trial_results["deciding_metric"])
        num_completed_folds.append(
            7
            if trial_results.get("pruned") is None or trial_results["pruned"] is False
            else trial_results["completed_folds"]
        )

    # Create a scatter plot of the results
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        trial_numbers,
        objective_values,
        c=num_completed_folds,
        cmap=cmaps.bubblegum,
        vmin=0,
        vmax=7,
        label="Objective Value",
    )

    # Add the colorbar corresponding to the scatter plot colors
    cbar = plt.colorbar(scatter)
    cbar.set_label("Number of Completed Folds", rotation=270, labelpad=15)

    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title("Study Results per Trial")
    plt.xlabel("Trial Number")
    plt.ylabel("Macro F1")
    plt.grid(True, alpha=0.3)

    # Save the plot to the output directory
    plot_path = outdir / "objective_value_per_trial.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    console.print(f"Plot saved to {plot_path}")


def plot_result_per_hparam(study_results, outdir):
    for hparam in ["lr", "weight_decay", "warmup_ratio", "dropout"]:
        # Extract the hyperparameter values and corresponding objective values
        hparam_values = []
        objective_values = []
        num_completed_folds = []
        for trial_results in study_results["trials"].values():
            hparam_values.append(trial_results["hparams"][hparam])
            objective_values.append(trial_results["deciding_metric"])
            num_completed_folds.append(
                7
                if trial_results.get("pruned") is None
                or trial_results["pruned"] is False
                else trial_results["completed_folds"]
            )

        if hparam == "weight_decay":
            hparam_values = [v if v != 0 else 1e-5 for v in hparam_values]

        # Create a scatter plot of the results
        plt.figure(figsize=(11, 6))
        scatter = plt.scatter(
            hparam_values,
            objective_values,
            c=num_completed_folds,
            cmap=cmaps.bubblegum,
            vmin=0,
            vmax=7,
            label="Objective Value",
        )

        # Add the colorbar corresponding to the scatter plot colors
        cbar = plt.colorbar(scatter)
        cbar.set_label("Number of Completed Folds", rotation=270, labelpad=15)

        if hparam == "lr":
            plt.xscale("log")
            plt.xticks(
                [5e-6, 1e-5, 3e-5, 5e-5],
                ["5e-6", "1e-5", "3e-5", "5e-5"],
            )
        if hparam == "weight_decay":
            plt.xscale("log")
            plt.xticks(
                [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
                ["0", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2", "1e-1"],
            )
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.title(f"Study Results per Hyperparameter: {hparam}")
        plt.xlabel(f"{hparam.replace('_', ' ').capitalize()} Value")
        plt.ylabel("Macro F1")
        plt.grid(True, alpha=0.3)

        # Save the plot to the output directory
        plot_path = outdir / f"objective_value_per_{hparam}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        console.print(f"Plot saved to {plot_path}")


def main():
    args = parse_args()
    args.input_file = Path(args.input_file).resolve()

    if not Path(args.input_file).exists():
        raise FileNotFoundError(f"Input not found: {args.input_file}")

    outdir = args.input_file.parent / "output"
    outdir.mkdir(parents=True, exist_ok=True)

    with args.input_file.open("r") as f:
        study_results = load(f)

    plot_result_per_trial(study_results, outdir)
    plot_result_per_hparam(study_results, outdir)


if __name__ == "__main__":
    main()
