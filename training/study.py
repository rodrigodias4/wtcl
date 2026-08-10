import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import signal

import numpy as np
import pandas as pd
import torch
import optuna

from study_plot import plot_result_per_hparam, plot_result_per_trial
from train import (
    PATIENCE,
    RANDOM_SEED,
    print_overall_results,
    set_random_seed,
    train_lodo,
)
from utils import (
    MODEL_DEFAULT,
    handle_interrupt,
    console,
    progress,
)
from rich.progress import TaskID

progress.speed_estimate_period = 60 * 60 * 6  # 6 hours
optuna.logging.set_verbosity(optuna.logging.WARNING)
datetime_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
progress_task_trials: TaskID = None
study_trials = {}
study_output_dir: Path = None

signal.signal(signal.SIGINT, handle_interrupt)

LR_RANGE = (5e-6, 5e-5)
WEIGHT_DECAY_OPTIONS = [0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1]
WARMUP_RATIO_RANGE = (0.0, 0.2)
DROPOUT_RANGE = (0.1, 0.4)
LR_FC_MULT_RANGE = (1, 50)
LR_CRF_MULT_RANGE = (1, 50)

# -----------------------------------------
# Hyperparameter tuning with Optuna
# -----------------------------------------


def load_model_as_first_trial(study: optuna.Study, model_path: str) -> None:
    model_path = Path(model_path)
    if not model_path.is_dir():
        console.print(
            f"Starting model path {model_path} does not exist or is not a directory. Ignoring."
        )
        return

    starting_hparams_path = model_path / "hyperparameters.json"
    if not starting_hparams_path.is_file():
        console.print(
            f"No hyperparameters.json found for the starting model at {starting_hparams_path}. Ignoring."
        )
        return
    starting_model_results_path = model_path / "results.json"
    if not starting_model_results_path.is_file():
        console.print(
            f"No results.json found for the starting model at {starting_model_results_path}. Ignoring."
        )
        return

    with open(model_path / "results.json", "r") as f:
        model_results = json.load(f)
        value = model_results["overall"]["validation"]["macro"]["f1"]
    with open(starting_hparams_path, "r") as f:
        starting_hparams = json.load(f)

    distributions = {
        "lr": optuna.distributions.FloatDistribution(*LR_RANGE, log=True),
        "weight_decay": optuna.distributions.CategoricalDistribution(
            WEIGHT_DECAY_OPTIONS
        ),
        "warmup_ratio": optuna.distributions.FloatDistribution(*WARMUP_RATIO_RANGE),
        "dropout": optuna.distributions.FloatDistribution(*DROPOUT_RANGE),
        "lr_fc_mult": optuna.distributions.FloatDistribution(
            *LR_FC_MULT_RANGE, log=True
        ),
        "lr_crf_mult": optuna.distributions.FloatDistribution(
            *LR_CRF_MULT_RANGE, log=True
        ),
    }

    params = {
        "lr": starting_hparams["lr"],
        "weight_decay": starting_hparams["weight_decay"],
        "warmup_ratio": starting_hparams["warmup_ratio"],
        "dropout": starting_hparams["dropout"],
        "lr_fc_mult": starting_hparams["lr_fc_mult"],
        "lr_crf_mult": starting_hparams["lr_crf_mult"],
    }

    """ intermediate_values = {
        i: v["best_validation_metrics"]["macro"]["f1"]
        for i, (k, v) in enumerate(model_results.items())
        if k != "overall" and "best_validation_metrics" in v
    } """

    intermediate_values = [
        model_results[k]["best_validation_metrics"]["macro"]["f1"]
        for k in sorted(model_results.keys())
        if k != "overall" and "best_validation_metrics" in model_results[k]
    ]

    intermediate_values = [
        float(np.mean(intermediate_values[: i + 1]))
        for i in range(len(intermediate_values))
    ]

    intermediate_values = {k: v for k, v in enumerate(intermediate_values)}

    first_trial = optuna.trial.create_trial(
        params=params,
        distributions=distributions,
        value=value,
        intermediate_values=intermediate_values,
        state=optuna.trial.TrialState.COMPLETE,
    )

    study.add_trial(first_trial)

    idx = max(study_trials.keys()) + 1 if study_trials else 0

    study_trials[idx] = {
        "hparams": starting_hparams,
        "results": model_results,
        "deciding_metric": value,
        "pruned": False,
    }

    console.print(
        f"Loaded starting model from '{model_path}' as trial {idx} with value {value:.1%} and intermediate values {list(intermediate_values.values())}"
    )

    save_study_results(study, study_output_dir)


# Hyperparameter sampling function
def sample_hparams(trial: optuna.Trial, crf: bool) -> dict:
    hparams = {
        "lr": trial.suggest_float("lr", *LR_RANGE, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", WEIGHT_DECAY_OPTIONS),
        "warmup_ratio": trial.suggest_float("warmup_ratio", *WARMUP_RATIO_RANGE),
        "dropout": trial.suggest_float("dropout", *DROPOUT_RANGE),
        "lr_fc_mult": trial.suggest_float(
            "lr_fc_mult",
            *LR_FC_MULT_RANGE,
            log=True,
        ),
    }

    if crf:
        hparams["lr_crf_mult"] = trial.suggest_float(
            "lr_crf_mult",
            *LR_CRF_MULT_RANGE,
            log=True,
        )

    return hparams


def save_study_results(study, outdir):
    study_results = {
        "study_name": study.study_name,
        "pruner": {
            "name": study.pruner.__class__.__name__,
            "n_startup_trials": getattr(study.pruner, "_n_startup_trials", None),
            "n_warmup_steps": getattr(study.pruner, "_n_warmup_steps", None),
        },
        "best_trial": (
            {
                "hparams": study.best_trial.params,
                "value": study.best_trial.value,
                "number": study.best_trial.number,
                "results": study_trials[study.best_trial.number]["results"]["overall"][
                    "validation"
                ],
            }
            if study.best_trial is not None
            else None
        ),
        "trials": study_trials,
    }
    with (outdir / "study_results.json").open("w") as f:
        json.dump(study_results, f, indent=4)

    return study_results


def callback(study: optuna.Study, trial: optuna.Trial):
    console.print(
        f"\nTrial {trial.number} finished with value: {f'{trial.value:.1%}' if trial.value is not None else 'n/a'}"
    )

    try:
        console.print(
            f"Best trial so far: {study.best_trial.number} with value: {study.best_trial.value:.1%}"
        )
    except ValueError:
        console.print("No best trial yet.")

    # Save the study results after each trial to ensure progress is not lost
    save_study_results(study, study_output_dir)

    # Advance the progress bar for hyperparameter tuning trials
    progress.advance(progress_task_trials)


def objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    model_name: str,
    fixed_hparams: dict,
) -> float:
    hparams = sample_hparams(trial, crf=fixed_hparams.get("use_crf", True))
    hparams = hparams | fixed_hparams  # Merge sampled hyperparameters with fixed ones

    console.rule(f"Trial {trial.number}", style="bold cyan")

    try:
        # Train the model with the current set of hyperparameters and get the fold metrics
        results, _, _, _, _ = train_lodo(
            df=df,
            model_name=model_name,
            hparams=hparams,
            val=True,
            model_output_dir=None,
            trial=trial,
        )

        print_overall_results(results)

        # Compute the deciding metric (average macro F1 across all debates) for this trial
        deciding_metric = results["overall"]["validation"]["macro"]["f1"]

        # Store the trial results in the global study_trials dictionary
        study_trials[trial.number] = {
            "hparams": hparams,
            "results": results,
            "deciding_metric": deciding_metric,
            "pruned": False,
        }

        return deciding_metric
    except optuna.exceptions.TrialPruned:
        # If the trial is pruned, store the partial trial data in the global study_trials dictionary
        partial_trial_data = trial.user_attrs.get("partial_trial_data")
        if partial_trial_data is not None:
            study_trials[trial.number] = partial_trial_data
        else:
            study_trials[trial.number] = {
                "hparams": hparams,
                "results": {},
                "deciding_metric": None,
                "pruned": True,
            }
        raise
    finally:
        # Free up memory after each trial
        gc.collect()
        torch.cuda.empty_cache()


# ------------------------------------------
# Main functions to run hyperparameter tuning
# ------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune hyperparameters for the model.")
    parser.add_argument("input_file", type=str, help="Path to input CSV file")
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_DEFAULT,
        help="Name of the model to use",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=20,
        help="Number of trials for hyperparameter tuning",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument(
        "--batch-size", type=int, required=True, help="Batch size for training"
    )
    parser.add_argument(
        "--no-crf",
        action="store_true",
        help="Whether to use a CRF layer on top of the transformer model.",
    )
    parser.add_argument(
        "--starting-hparams",
        type=str,
        default=None,
        help="Path to JSON file with starting hyperparameters for the study.",
    )
    parser.add_argument(
        "--starting-model",
        type=str,
        default=None,
        help="Path to a pre-trained model folder to set as the first trial in the study.",
    )
    parser.add_argument(
        "--num-startup-trials",
        type=int,
        default=5,
        help="Number of startup trials for the Optuna pruner.",
    )
    parser.add_argument(
        "--num-warmup-steps",
        type=int,
        default=2,
        help="Number of warmup steps (folds) for the Optuna pruner.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="median",
        choices=["median", "threshold", "none"],
        help="Type of pruner to use for Optuna.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a journal file to resume the study from a previous run.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory during training.",
    )
    # TODO: Add additional hyperparameters
    args = parser.parse_args()
    return args


def main():
    global progress_task_trials, study_output_dir
    args = parse_args()
    set_random_seed(args.seed)
    study_name = f"study_{args.input_file.split('/')[-1].split('.')[0].split('_')[0]}_{args.model_name.split('/')[-1]}_{datetime_now}"
    fixed_hparams = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "use_crf": not args.no_crf,
        "crf_priors": False,
        "emission_bias": False,
        "freeze": 0,
        "seed": args.seed,
        "patience": PATIENCE,
        "mixed_precision_dtype": "bf16",
        "gradient_checkpointing": args.gradient_checkpointing,
    }

    dataset_name = Path(args.input_file).name.split(".")[0].split("_")[0]

    if args.resume_from:
        console.print(f"Resuming study from {args.resume_from}")
        study_output_dir = (
            Path(args.resume_from).parent
            if Path(args.resume_from).is_file()
            else Path(args.resume_from)
        )
        study_output_path = study_output_dir / "study_results.json"
        if not study_output_path.is_file():
            console.print(
                f"No study_results.json found in {study_output_dir}. Starting fresh."
            )
        else:
            console.print(f"Loading existing study results from {study_output_path}")
            with study_output_path.open("r") as f:
                existing_results = json.load(f)
            for trial_number, trial_data in existing_results["trials"].items():
                study_trials[int(trial_number)] = trial_data

            study_name = existing_results.get("study_name", study_name)
            args.num_startup_trials = existing_results.get("pruner", {}).get(
                "n_startup_trials", args.num_startup_trials
            )
            args.num_warmup_steps = existing_results.get("pruner", {}).get(
                "n_warmup_steps", args.num_warmup_steps
            )
    else:
        # Create output directory for the study results
        study_output_dir = (
            Path(os.path.dirname(os.path.abspath(__file__)))
            / "studies"
            / dataset_name
            / args.model_name.split("/")[-1]
            / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        )

        study_output_dir.mkdir(exist_ok=True, parents=True)
        study_output_path = study_output_dir / "study_results.json"

    file_backend = optuna.storages.journal.JournalFileBackend(
        str(Path(args.resume_from))
        if args.resume_from and Path(args.resume_from).is_file()
        else str(study_output_dir / "study.journal")
    )
    storage = optuna.storages.JournalStorage(file_backend)
    console.print(
        f"Study journal will be saved to {study_output_dir / 'study.journal'}"
    )
    console.print(
        f"Study results will be saved to {study_output_dir / 'study_results.json'}"
    )

    df = pd.read_csv(args.input_file)
    console.print(f"Loaded {len(df)} rows from {args.input_file}")

    # Start the progress bar for hyperparameter tuning trials
    progress_task_trials = progress.add_task(
        f"Hyperparameter Tuning Trials", total=args.num_trials
    )
    progress.start()

    # Create the Optuna pruner based on the specified type
    match args.pruner:
        case "median":
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=args.num_startup_trials,
                n_warmup_steps=args.num_warmup_steps,
            )
        case "threshold":
            pruner = optuna.pruners.ThresholdPruner(
                lower=0.65,
                n_warmup_steps=args.num_warmup_steps,
            )
        case "none":
            pruner = optuna.pruners.NopPruner()
        case _:
            raise ValueError(f"Unsupported pruner type: {args.pruner}")

    console.print(
        f"Using {pruner.__class__.__name__} for pruning trials: "
        f"N startup trials: {getattr(pruner, '_n_startup_trials', 'N/A')}, "
        f"N warmup steps: {getattr(pruner, '_n_warmup_steps', 'N/A')}"
    )

    # Create a new Optuna study for hyperparameter tuning
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    if args.resume_from:
        console.print(
            f"Loaded study '{study.study_name}' with {len(study.get_trials(deepcopy=False))} existing trials."
        )
        if study.best_trial is not None:
            console.print(
                f"Best trial so far: {study.best_trial.number} with value: {study.best_trial.value:.1%}"
            )

    if args.starting_model:
        # Load the starting model's results and hyperparameters and add it as the first trial in the study
        load_model_as_first_trial(study, args.starting_model)

    if args.starting_hparams:
        with open(args.starting_hparams, "r") as f:
            starting_hparams = json.load(f)
        study.enqueue_trial(starting_hparams)

    # Run hyperparameter tuning with Optuna
    study.optimize(
        lambda trial: objective(
            trial,
            df,
            args.model_name,
            fixed_hparams=fixed_hparams,
        ),
        n_trials=args.num_trials,
        show_progress_bar=False,
        callbacks=[callback],
    )

    # Stop the progress bar for hyperparameter tuning trials
    progress.remove_task(progress_task_trials)
    progress.stop()

    study_results = save_study_results(study, study_output_dir)
    console.print(f"Study results saved to {study_output_dir / 'study_results.json'}")

    console.rule(f"Study Results")
    console.print(f"Best trial number: {study.best_trial.number}")
    console.print(f"Best trial value (average macro F1): {study.best_trial.value:.4f}")
    console.print("Best hyperparameters:")
    [
        console.print(f"‣ {key}: {value}")
        for key, value in study.best_trial.params.items()
    ]

    # Plot the results of the study
    (study_output_dir / "output").mkdir(parents=True, exist_ok=True)
    plot_result_per_trial(study_results, study_output_dir / "output")
    plot_result_per_hparam(study_results, study_output_dir / "output")

    del study  # Free up memory


if __name__ == "__main__":
    main()
