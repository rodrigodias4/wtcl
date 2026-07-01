import argparse
from datetime import datetime
import gc
import json
import os
import signal

import pandas as pd
import torch
import optuna

from train import (
    RANDOM_SEED,
    print_overall_results,
    set_random_seed,
    train_lodo,
    script_dir,
)
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

progress.speed_estimate_period = 60 * 60 * 6  # 6 hours
optuna.logging.set_verbosity(optuna.logging.WARNING)
datetime_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
study_trials = {}
progress_task_trials: TaskID = None

signal.signal(signal.SIGINT, handle_interrupt)

# -----------------------------------------
# Hyperparameter tuning with Optuna
# -----------------------------------------


# Hyperparameter sampling function
def sample_hparams(trial: optuna.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-6, 5e-5, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8]),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1]
        ),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
    }


def callback(study: optuna.Study, trial: optuna.Trial):
    console.print(f"\nTrial {trial.number} finished with value: {trial.value:.1%}")
    console.print(
        f"Best trial so far: {study.best_trial.number} with value: {study.best_trial.value:.1%}"
    )
    # Advance the progress bar for hyperparameter tuning trials
    progress.advance(progress_task_trials)


def objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    model_name: str,
    fixed_hparams: dict,
) -> float:
    hparams = sample_hparams(trial)
    hparams = hparams | fixed_hparams  # Merge sampled hyperparameters with fixed ones

    console.rule(f"Trial {trial.number}", style="bold cyan")

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
    }

    # Free up memory after each trial
    gc.collect()
    torch.cuda.empty_cache()

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
        "--num_epochs", type=int, default=15, help="Number of epochs for training"
    )
    parser.add_argument(
        "--use_crf",
        type=bool,
        default=True,
        help="Whether to use a CRF layer on top of the transformer model.",
    )
    parser.add_argument(
        "--starting_hparams",
        type=str,
        default=None,
        help="Path to JSON file with starting hyperparameters for the study.",
    )
    parser.add_argument(
        "--n_startup_trials",
        type=int,
        default=5,
        help="Number of startup trials for the Optuna pruner.",
    )
    parser.add_argument(
        "--n_warmup_steps",
        type=int,
        default=2,
        help="Number of warmup steps (folds) for the Optuna pruner.",
    )
    # TODO: Add additional hyperparameters
    args = parser.parse_args()
    return args


def main():
    global progress_task_trials
    set_random_seed(RANDOM_SEED)
    args = parse_args()
    study_name = f"study_{args.input_file.split('/')[-1].split('.')[0].split('_')[0]}_{args.model_name.split('/')[-1]}_{datetime_now}"

    fixed_hparams = {
        "num_epochs": args.num_epochs,
        "use_crf": True,
        "debate_alpha": DEBATE_TEMPERED_SAMPLING_ALPHA,
        "bio_alpha": BIO_TEMPERED_SAMPLING_ALPHA,
        "bio_eps": BIO_TEMPERED_SAMPLING_EPS,
        "crf_priors": False,
        "emission_bias": False,
        "freeze": 0,
        "mixed_precision": True,
    }

    # Create output directory for the study results
    study_output_path = script_dir / "studies" / f"{study_name}.json"
    os.makedirs(study_output_path.parent, exist_ok=True)

    df = pd.read_csv(args.input_file)
    console.print(f"Loaded {len(df)} rows from {args.input_file}")

    # Start the progress bar for hyperparameter tuning trials
    progress_task_trials = progress.add_task(
        f"Hyperparameter Tuning Trials", total=args.num_trials
    )
    progress.start()

    # Create a new Optuna study for hyperparameter tuning
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=args.n_startup_trials, n_warmup_steps=args.n_warmup_steps
        ),
        study_name=study_name,
    )

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

    with open(study_output_path, "w") as f:
        json.dump(
            {
                "best_trial": {
                    "hparams": study.best_trial.params,
                    "value": study.best_trial.value,
                    "number": study.best_trial.number,
                    "results": study_trials[study.best_trial.number]["results"][
                        "overall"
                    ]["validation"],
                },
                "trials": study_trials,
            },
            f,
            indent=4,
        )

    console.rule(f"Study Results")
    console.print(f"Best trial number: {study.best_trial.number}")
    console.print(f"Best trial value (average macro F1): {study.best_trial.value:.4f}")
    console.print("Best hyperparameters:")
    [
        console.print(f"‣ {key}: {value}")
        for key, value in study.best_trial.params.items()
    ]

    del study  # Free up memory


if __name__ == "__main__":
    main()
