import argparse
import datetime
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna
from transformers import get_linear_schedule_with_warmup

from train import (
    MAX_LENGTH,
    MODEL_DEFAULT,
    RANDOM_SEED,
    WTCLDataset,
    build_model,
    compute_metrics_token_level,
    evaluate,
    get_device,
    get_model_output_dir,
    get_tokenizer,
    set_random_seed,
    train,
    train_lodo,
    label2id,
    script_dir,
)
from plot import plot_train_val_loss_curves, plot_metric_curves


def get_validation_debate(df: pd.DataFrame, debates: list[str]) -> str:
    """
    Choosing validation debate that minimizes the distance to the mean
    ratio of positive tokens across all debates.
    This is a heuristic to select a validation debate that is representative of the overall dataset.
    """
    debate_ratios = {}
    for debate in debates:
        n_positive_tokens = 0
        n_tokens = 0
        debate_data = df[df["debate_id"] == debate]

        for index, row in debate_data.iterrows():
            labels = eval(row["labels"])
            n_positive_tokens += sum(label2id[label] > 0 for label in labels)
            n_tokens += len(labels)

        debate_ratios[debate] = n_positive_tokens / n_tokens if n_tokens > 0 else 0

    # Find the debate with the closest number of positive tokens to the mean
    mean_positive_tokens = sum(debate_ratios.values()) / len(debate_ratios)
    val_debate = min(
        debate_ratios, key=lambda x: abs(debate_ratios[x] - mean_positive_tokens)
    )

    return val_debate


# -----------------------------------------
# Hyperparameter tuning with Optuna
# -----------------------------------------


# Hyperparameter sampling function
def sample_hparams(trial: optuna.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 5e-5, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "dropout": trial.suggest_float("dropout", 0.1, 0.3),
    }


def objective(
    trial: optuna.Trial, df: pd.DataFrame, model_name: str, fixed_hparams: dict = None
) -> float:
    hparams = sample_hparams(trial)

    fold_metrics = []
    step = 0
    debates = df["debate_id"].unique()
    tokenizer = get_tokenizer(model_name)

    for test_debate in tqdm(debates, desc="Leave-One-Debate-Out Folds"):
        tqdm.write(
            f"Tuning models with {test_debate} left out as test debate and hyperparameters: {hparams}"
        )

        train_val_debates = list([d for d in debates if d != test_debate])
        val_debate = get_validation_debate(df, train_val_debates)
        train_debates = [d for d in train_val_debates if d != val_debate]

        # Prepare datasets and dataloaders
        train_data = df[df["debate_id"].isin(train_debates)]
        val_data = df[df["debate_id"] == val_debate]
        tqdm.write(f"Validating on debate: {val_debate} | Train size: {len(train_data)} | Val size: {len(val_data)}")
        train_dataset = WTCLDataset(
            train_data.to_dict("records"), tokenizer, MAX_LENGTH
        )
        val_dataset = WTCLDataset(
            val_data.to_dict("records"), tokenizer, MAX_LENGTH
        )
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

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"]
        )
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

        preds, labels, _ = evaluate(model, val_loader, get_device())

        metrics = compute_metrics_token_level(preds, labels)
        fold_metrics.append(metrics)
        trial.report(metrics["macro"]["f1"], step)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        step += 1

    return sum([metrics["macro"]["f1"] for metrics in fold_metrics]) / len(fold_metrics)


# ------------------------------------------
# Results processing and plotting calls
# ------------------------------------------


def process_results(results: dict, best_params: dict) -> None:
    with (
        script_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S") / "results.json"
    ).open("w") as f:
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
        "--num_epochs", type=int, default=5, help="Number of epochs for training"
    )
    args = parser.parse_args()
    return args


def main():
    set_random_seed(RANDOM_SEED)

    args = parse_args()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(),
    )

    df = pd.read_csv(args.input_file)
    print(f"Loaded {len(df)} rows from {args.input_file}")

    study.optimize(
        lambda trial: objective(
            trial, df, args.model_name, fixed_hparams={"num_epochs": args.num_epochs}
        ),
        n_trials=args.num_trials,
        show_progress_bar=True,
    )
    best_params = study.best_trial.params
    print("Best hyperparameters:", best_params)

    # Train final model with best hyperparameters on the full dataset
    model_output_dir = get_model_output_dir("final_model")
    model_output_dir.mkdir(exist_ok=True, parents=True)
    results, models = train_lodo(df, args.model_name, best_params, model_output_dir)
    process_results(results, best_params)


if __name__ == "__main__":
    main()
