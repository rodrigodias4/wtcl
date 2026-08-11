from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
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
        trial_numbers.append(int(trial_number))
        objective_values.append(trial_results["deciding_metric"])
        num_completed_folds.append(
            7
            if trial_results.get("pruned") is None or trial_results["pruned"] is False
            else trial_results["completed_folds"]
        )
    max_objective_value_at_trial = []
    best_trial_numbers = []
    best_objective_values = []
    running_max = float("-inf")
    for trial_number, objective_value in zip(trial_numbers, objective_values):
        if (
            num_completed_folds[trial_numbers.index(trial_number)] == 7
            and objective_value > running_max
        ):
            best_trial_numbers.append(trial_number)
            best_objective_values.append(objective_value)
            running_max = objective_value
        max_objective_value_at_trial.append(running_max)

    # Create a scatter plot of the results
    plt.figure(figsize=(max(10, max(trial_numbers) * 0.225), 6))
    scatter = plt.scatter(
        trial_numbers,
        objective_values,
        s=25,
        c=num_completed_folds,
        cmap=cmaps.bubblegum.reversed(),
        vmin=study_results.get("pruner", {}).get("n_warmup_steps", 0) + 1,
        vmax=7,
        label="Objective Value",
        zorder=2,
    )

    # Add a line plot for the maximum objective value at each trial
    max_at_trial = plt.step(
        trial_numbers,
        max_objective_value_at_trial,
        where="post",
        color="blue",
        alpha=0.2,
        linestyle=(0, (4, 4)),
        linewidth=0.9,
        label="Running Max Objective Value",
        zorder=1,
    )

    best_trials = None
    if best_trial_numbers:
        best_trials = plt.scatter(
            best_trial_numbers,
            best_objective_values,
            marker="D",
            s=40,
            facecolors=cmaps.bubblegum.reversed()(1.0),
            edgecolors="white",
            linewidth=0.5,
            label="New Best Trial",
            zorder=4,
        )

    # Add the colorbar corresponding to the scatter plot colors
    cbar = plt.colorbar(scatter)
    cbar.set_label("Number of Completed Folds", rotation=270, labelpad=15)
    cbar.set_ticks(np.arange(cbar.vmin, cbar.vmax + 1, 1))

    # Shade startup trials region
    n_startup_trials = study_results.get("pruner", {}).get("n_startup_trials", 0)
    if n_startup_trials > 0:
        plt.plot(
            [0, n_startup_trials - 1],
            [0.2, 0.2],
            color="cornflowerblue",
            markersize=0,
        )
        plt.plot([0, 0], [0.19, 0.21], color="cornflowerblue", markersize=0)
        plt.plot(
            [n_startup_trials - 1, n_startup_trials - 1],
            [0.19, 0.21],
            color="cornflowerblue",
            markersize=0,
        )
        plt.text(
            (n_startup_trials - 1) / 2,
            0.17,
            "Startup Trials",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
            color="black",
        )

    # Shade random trials region
    plt.plot(
        [0, 9],
        [0.1, 0.1],
        color="violet",
        markersize=0,
    )
    plt.plot([0, 0], [0.09, 0.11], color="violet", markersize=0)
    plt.plot([9, 9], [0.09, 0.11], color="violet", markersize=0)
    plt.text(
        4.5,
        0.07,
        "Random Trials",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=8,
        color="black",
    )

    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    # plt.title("Study Results per Trial")
    plt.xlabel("Trial Number (0-indexed)")
    plt.gca().xaxis.set_major_locator(
        plt.MaxNLocator(integer=True, nbins=24, steps=[1, 2, 5, 10], prune=None)
    )
    plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
    plt.ylabel("Validation Macro-F1")
    plt.grid(True, alpha=0.2, zorder=0)
    legend_handles = [max_at_trial[0]]
    if best_trials is not None:
        legend_handles.append(best_trials)
    plt.legend(handles=legend_handles, loc="lower right")

    # Save the plot to the output directory
    plot_path = outdir / "objective_value_per_trial.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    console.print(f"Plot saved to {plot_path}")


def plot_result_per_hparam(study_results, outdir):
    for hparam in [
        "lr",
        "weight_decay",
        "warmup_ratio",
        "dropout",
        "lr_fc_mult",
        "lr_crf_mult",
    ]:
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
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            hparam_values,
            objective_values,
            c=num_completed_folds,
            cmap=cmaps.bubblegum.reversed(),
            vmin=study_results.get("pruner", {}).get("n_warmup_steps", 0) + 1,
            vmax=7,
            s=64,
            edgecolor="white",
            linewidth=0.5,
            label="Objective Value",
            zorder=2,
        )

        # Add the colorbar corresponding to the scatter plot colors
        cbar = plt.colorbar(scatter)
        cbar.set_label("Number of Completed Folds", rotation=270, labelpad=15)
        cbar.set_ticks(np.arange(cbar.vmin, cbar.vmax + 1, 1))

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
        # plt.title(f"Study Results per Hyperparameter: {hparam}")
        plt.xlabel(f"{hparam.replace('_', ' ').capitalize()} Value")
        plt.ylabel("Macro F1")
        plt.grid(True, alpha=0.3, zorder=0)

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
