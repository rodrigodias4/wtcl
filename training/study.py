import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import signal

import pandas as pd
import torch
import optuna

from study_analysis import plot_result_per_hparam, plot_result_per_trial
from train import (
    PATIENCE,
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
progress_task_trials: TaskID = None
study_trials = {}
study_output_dir: Path = None

signal.signal(signal.SIGINT, handle_interrupt)

# -----------------------------------------
# Hyperparameter tuning with Optuna
# -----------------------------------------


# Hyperparameter sampling function
def sample_hparams(trial: optuna.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 5e-6, 5e-5, log=True),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1]
        ),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "lr_fc_mult": trial.suggest_float(
            "lr_fc_mult",
            1,
            50,
            log=True,
        ),
        "lr_crf_mult": trial.suggest_float(
            "lr_crf_mult",
            1,
            50,
            log=True,
        ),
    }


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
    hparams = sample_hparams(trial)
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
        "--use-crf",
        type=bool,
        default=True,
        help="Whether to use a CRF layer on top of the transformer model.",
    )
    parser.add_argument(
        "--starting-hparams",
        type=str,
        default=None,
        help="Path to JSON file with starting hyperparameters for the study.",
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
        "use_crf": True,
        "debate_alpha": DEBATE_TEMPERED_SAMPLING_ALPHA,
        "bio_alpha": BIO_TEMPERED_SAMPLING_ALPHA,
        "bio_eps": BIO_TEMPERED_SAMPLING_EPS,
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
        study_output_dir = Path(args.resume_from).parent
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
    plot_result_per_trial(study_results, study_output_dir / "output")
    plot_result_per_hparam(study_results, study_output_dir / "output")

    del study  # Free up memory


if __name__ == "__main__":
    main()
