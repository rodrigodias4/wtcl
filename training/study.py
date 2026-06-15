import argparse
import datetime
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna
from transformers import get_linear_schedule_with_warmup

from train import (
    MAX_LENGTH,
    RANDOM_SEED,
    WTCLDataset,
    build_model,
    get_device,
    get_model_output_dir,
    get_tokenizer,
    set_random_seed,
    train,
    train_lodo,
    script_dir,
)
from plot import plot_train_val_loss_curves, plot_metric_curves
from utils import (
    get_optimizer,
    get_validation_debate,
    MODEL_DEFAULT,
)

datetime_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
    fixed_hparams: dict = None,
    study_output_path: Path = None,
) -> float:
    hparams = sample_hparams(trial)

    fold_metrics = []
    step = 0
    debates = df["debate_id"].unique()
    tokenizer = get_tokenizer(model_name)
    tqdm.write(f"Trial {trial.number} with hyperparameters: {hparams}")
    for test_debate in tqdm(debates, desc="Leave-One-Debate-Out Folds"):
        tqdm.write(f"Tuning models with {test_debate} left out as test debate")

        train_val_debates = list([d for d in debates if d != test_debate])
        val_debate = get_validation_debate(df, train_val_debates)
        train_debates = [d for d in train_val_debates if d != val_debate]

        # Prepare datasets and dataloaders
        train_data = df[df["debate_id"].isin(train_debates)]
        val_data = df[df["debate_id"] == val_debate]
        assert (
            not val_data["debate_id"].isin(train_debates).any()
        ), "Validation debate should not be in training debates"
        tqdm.write(
            f"Validating on debate: {val_debate} | Train size: {len(train_data)} | Val size: {len(val_data)}"
        )
        train_dataset = WTCLDataset(
            train_data.to_dict("records"), tokenizer, MAX_LENGTH
        )
        val_dataset = WTCLDataset(val_data.to_dict("records"), tokenizer, MAX_LENGTH)
        train_loader = DataLoader(
            train_dataset, batch_size=hparams["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=hparams["batch_size"], shuffle=False
        )
        del (
            train_data,
            train_dataset,
            val_dataset,
            val_debate,
            train_debates,
            train_val_debates,
        )  # Free up memory

        model = build_model(model_name, hparams)
        optimizer = get_optimizer(model, hparams)
        total_steps = len(train_loader) * fixed_hparams["num_epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(hparams["warmup_ratio"] * total_steps),
            num_training_steps=total_steps,
        )

        results = train(
            model,
            train_loader,
            optimizer,
            scheduler,
            get_device(),
            fixed_hparams["num_epochs"],
            val_loader,
        )
        fold_metrics.append(results)
        trial.report(results["validation_metrics"][-1]["macro"]["f1"], step)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        step += 1

        gc.collect()
        torch.cuda.empty_cache()

    deciding_metric = sum(
        [metrics["validation_metrics"][-1]["macro"]["f1"] for metrics in fold_metrics]
    ) / len(fold_metrics)

    with study_output_path.open("a") as f:
        json.dump(
            {
                "trial_number": trial.number,
                "hparams": hparams,
                "fold_metrics": fold_metrics,
                "deciding_metric": deciding_metric,
            },
            f,
        )
        f.write("\n")

    gc.collect()
    torch.cuda.empty_cache()
    return deciding_metric


# ------------------------------------------
# Results processing and plotting calls
# ------------------------------------------


def process_results(results: dict, best_params: dict, study_output_path: Path) -> None:
    with (study_output_path).open("a") as f:
        json.dump({"best_params": best_params, "results": results}, f, indent=4)

    plot_train_val_loss_curves(results)
    plot_metric_curves(results)


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
    args = parser.parse_args()
    return args


def main():
    set_random_seed(RANDOM_SEED)

    args = parse_args()

    study_name = f"study_{args.input_file.name.split('.')[0].split('_')[0]}_{args.model_name}_{datetime_now}"

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(),
        study_name=study_name,
    )
    study_output_path = script_dir / f"{study_name}.json"
    with study_output_path.open("w") as f:
        pass  # Create empty file

    df = pd.read_csv(args.input_file)
    print(f"Loaded {len(df)} rows from {args.input_file}")

    study.optimize(
        lambda trial: objective(
            trial,
            df,
            args.model_name,
            fixed_hparams={"num_epochs": args.num_epochs},
            study_output_path=study_output_path,
        ),
        n_trials=args.num_trials,
        show_progress_bar=True,
    )
    best_params = study.best_trial.params.copy()
    print("Best hyperparameters:", best_params)

    del study  # Free up memory

    # Train final model with best hyperparameters on the full dataset
    model_output_dir = get_model_output_dir("final_model")
    model_output_dir.mkdir(exist_ok=True, parents=True)
    final_hparams = best_params | {"num_epochs": args.num_epochs}
    results = train_lodo(df, args.model_name, final_hparams, True, model_output_dir)
    process_results(results, best_params)


if __name__ == "__main__":
    main()
