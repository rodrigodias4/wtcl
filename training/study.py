import argparse
import datetime
import gc
import json
import os
from pathlib import Path
import signal

import pandas as pd
import torch
import optuna

from train import (
    RANDOM_SEED,
    get_model_output_dir,
    set_random_seed,
    train_lodo,
    script_dir,
)
from plot import plot_train_val_loss_curves, plot_metric_curves
from training.plot_cm import plot_confusion_matrix
from utils import (
    BIO_TEMPERED_SAMPLING_ALPHA,
    BIO_TEMPERED_SAMPLING_EPS,
    DEBATE_TEMPERED_SAMPLING_ALPHA,
    MODEL_DEFAULT,
    handle_interrupt,
    console,
    progress,
)
from rich.progress import TaskID

datetime_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
study_trials = {}

signal.signal(signal.SIGINT, handle_interrupt)

# -----------------------------------------
# Hyperparameter tuning with Optuna
# -----------------------------------------


# Hyperparameter sampling function
def sample_hparams(trial: optuna.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 5e-5, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "dropout": trial.suggest_float("dropout", 0.2, 0.4),
    }


def objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    model_name: str,
    fixed_hparams: dict,
    progress_task_trials: TaskID = None,
) -> float:
    hparams = sample_hparams(trial)
    hparams = hparams | fixed_hparams  # Merge sampled hyperparameters with fixed ones

    console.rule(f"Trial {trial.number}")

    # Train the model with the current set of hyperparameters and get the fold metrics
    fold_metrics, _ = train_lodo(df, model_name, hparams, False, None, False, trial)

    # Compute the deciding metric (average macro F1 across all debates) for this trial
    deciding_metric = sum(
        [
            metrics["validation_metrics"][metrics["best_epoch"] - 1]["macro"]["f1"]
            for metrics in fold_metrics
        ]
    ) / len(fold_metrics)

    # Store the trial results in the global study_trials dictionary
    study_trials[trial.number] = {
        "hparams": hparams,
        "fold_metrics": fold_metrics,
        "deciding_metric": deciding_metric,
    }

    # Free up memory after each trial
    gc.collect()
    torch.cuda.empty_cache()

    # Advance the progress bar for hyperparameter tuning trials
    if progress_task_trials is not None:
        progress.update(progress_task_trials, advance=1)

    return deciding_metric

# ------------------------------------------
# Main functions to run hyperparameter tuning
# ------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune hyperparameters for the model.")
    parser.add_argument("input_file", type=str, help="Path to input CSV file")
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_DEFAULT,
        help="Name of the model to use",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=20,
        help="Number of trials for hyperparameter tuning",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=8, help="Number of epochs for training"
    )
    parser.add_argument(
        "--use_crf",
        type=bool,
        default=True,
        help="Whether to use a CRF layer on top of the transformer model.",
    )
    # TODO: Add additional hyperparameters
    args = parser.parse_args()
    return args


def main():
    set_random_seed(RANDOM_SEED)
    args = parse_args()
    study_name = f"study_{args.input_file.name.split('.')[0].split('_')[0]}_{args.model_name.split("/")[-1]}_{datetime_now}"

    # Create a new Optuna study for hyperparameter tuning
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(),
        study_name=study_name,
    )

    fixed_hparams = {
        "use_crf": True,
        "debate_alpha": DEBATE_TEMPERED_SAMPLING_ALPHA,
        "bio_alpha": BIO_TEMPERED_SAMPLING_ALPHA,
        "bio_eps": BIO_TEMPERED_SAMPLING_EPS,
        "crf_priors": False,
        "emission_bias": False,
        "freeze": 0,
    }

    # Create output directory for the study results
    study_output_path = script_dir / "studies" / f"{study_name}.json"
    os.makedirs(study_output_path.parent, exist_ok=True, parents=True)

    df = pd.read_csv(args.input_file)
    console.print(f"Loaded {len(df)} rows from {args.input_file}")

    # Start the progress bar for hyperparameter tuning trials
    progress_task_trials = progress.add_task(
        f"Hyperparameter Tuning Trials", total=args.num_trials
    )
    progress.start()

    # Run hyperparameter tuning with Optuna
    study.optimize(
        lambda trial: objective(
            trial,
            df,
            args.model_name,
            fixed_hparams=fixed_hparams,
            progress_task_trials=progress_task_trials,
        ),
        n_trials=args.num_trials,
        show_progress_bar=True,
    )

    # Stop the progress bar for hyperparameter tuning trials
    progress.remove_task(progress_task_trials)
    progress.stop()

    console.rule(f"Study Results")
    console.print(f"Best trial number: {study.best_trial.number}")
    console.print(f"Best trial value (average macro F1): {study.best_trial.value:.4f}")
    console.print("Best hyperparameters:")
    [console.print(f"‣ {key}: {value}") for key, value in study.best_trial.params.items()]

    del study  # Free up memory


if __name__ == "__main__":
    main()
