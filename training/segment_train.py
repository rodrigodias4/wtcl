import json
from contextlib import nullcontext
from argparse import ArgumentParser
from pathlib import Path
import time

from optuna import TrialPruned
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModel, get_linear_schedule_with_warmup
import numpy as np

from train import (
    EARLY_STOPPING_DELTA,
    PATIENCE,
    RANDOM_SEED,
    get_device,
    get_model_output_dir,
    get_tokenizer,
    get_validation_debate,
    mp_str_to_dtype,
    set_random_seed,
)
from utils import console, progress

MAX_LENGTH = 128


class SegmentDataset(Dataset):
    """Tokenized text segments with a binary whole-segment label."""

    def __init__(self, data, tokenizer, max_length, label_column="label"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_column = label_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        label = item[self.label_column]
        if pd.isna(label):
            raise ValueError(f"Missing binary label at dataset row {index}")

        encoding = self.tokenizer(
            str(item["text"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(label), dtype=torch.long),
        }


class SegmentClassifier(nn.Module):
    """Transformer encoder with a binary whole-segment classification head."""

    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        segment_representation = output.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(segment_representation))


def parse_args():
    parser = ArgumentParser(
        description="Train a transformer binary classifier for annotated segments."
    )
    parser.add_argument("dataset_path", type=str, help="Path to the dataset CSV.")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="FacebookAI/roberta-base",
        help="Hugging Face transformer checkpoint to use.",
    )
    parser.add_argument(
        "--hparams-path",
        type=str,
        help="Optional JSON file containing hyperparameter overrides.",
    )
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", "--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument(
        "--mixed-precision-dtype",
        choices=["fp16", "bf16", "none"],
        default="bf16",
    )
    parser.add_argument(
        "--val",
        action="store_true",
    )
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument(
        "--save",
        action="store_true",
    )
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def load_data(path, label_column="label"):
    dataframe = pd.read_csv(path)
    if "text" not in dataframe or label_column not in dataframe:
        raise ValueError(
            f"Dataset must contain 'text' and {label_column!r} columns; "
            f"found {list(dataframe.columns)}"
        )
    dataframe[label_column] = pd.to_numeric(dataframe[label_column], errors="raise")
    if not dataframe[label_column].isin([0, 1]).all():
        raise ValueError(f"{label_column!r} must contain only 0 and 1")
    if dataframe[label_column].nunique() < 2:
        raise ValueError("Training requires both binary classes")

    return dataframe


def metrics(predictions, labels):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses, predictions, labels = [], [], []
    with torch.inference_mode():
        for batch in dataloader:
            logits = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            batch_labels = batch["labels"].to(device)
            losses.append(criterion(logits, batch_labels).item() * len(batch_labels))
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
    return (
        sum(losses) / len(labels),
        metrics(predictions, labels),
        predictions,
        labels,
    )


def make_loader(dataframe, tokenizer, batch_size, label_column, shuffle):
    dataset = SegmentDataset(
        dataframe.to_dict("records"), tokenizer, MAX_LENGTH, label_column
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train(
    model,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epochs,
    val_loader=None,
    mixed_precision_dtype=None,
    patience=PATIENCE,
):
    """Train a segment classifier and return its history and best state."""
    use_scaler = mixed_precision_dtype == torch.float16 and device.type == "cuda"
    scaler = GradScaler(enabled=use_scaler)

    best_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    training_loss_history = []
    validation_loss_history = []
    validation_metrics_history = []
    best_epoch = 0
    best_validation_metrics = None
    progress_epochs = progress.add_task("Epochs", total=epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        training_losses = []
        progress_batches = progress.add_task("Batches", total=len(train_loader))
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            context = (
                autocast(device_type=device.type, dtype=mixed_precision_dtype)
                if mixed_precision_dtype is not None
                else nullcontext()
            )
            with context:
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                old_scale = scaler.get_scale()
                scaler.update()
                if old_scale <= scaler.get_scale():
                    scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                scheduler.step()

            training_losses.append(loss.item() * len(batch["labels"]))
            progress.advance(progress_batches)

        progress.remove_task(progress_batches)
        training_loss_history.append(sum(training_losses) / len(train_loader.dataset))
        progress.advance(progress_epochs)

        if val_loader is None:
            best_state = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            console.print(f"Epoch {epoch}: TL={training_loss_history[-1]*100:.2f}")
            continue

        validation_loss, validation_metrics, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        validation_loss_history.append(validation_loss)
        validation_metrics_history.append(validation_metrics)
        if validation_metrics["f1"] > best_f1 + EARLY_STOPPING_DELTA:
            best_f1 = validation_metrics["f1"]
            best_state = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            best_validation_metrics = validation_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        console.print(
            f"{('[magenta]░░ [/magenta]') if epoch == best_epoch else '░░ '}"
            f"Epoch {epoch}: "
            f"TL={training_loss_history[-1]*100:.2f} "
            f"VL={validation_loss*100:.2f} "
            f"F1={validation_metrics['f1']*100:.1f} "
            f"P={validation_metrics['precision']*100:.1f} "
            f"R={validation_metrics['recall']*100:.1f}"
        )

        if epochs_without_improvement >= patience:
            break

    progress.remove_task(progress_epochs)
    return (
        {
            "training_loss": training_loss_history,
            "validation_loss": validation_loss_history,
            "validation_metrics": validation_metrics_history,
            "best_epoch": best_epoch,
            "best_validation_metrics": best_validation_metrics,
        },
        best_state,
    )


def train_fold(
    train_data,
    validation_data,
    hparams,
    tokenizer,
    device,
    fold_dir,
    save=False,
):
    train_loader = make_loader(
        train_data,
        tokenizer,
        hparams["batch_size"],
        "label",
        shuffle=True,
    )
    validation_loader = (
        make_loader(
            validation_data,
            tokenizer,
            hparams["batch_size"],
            "label",
            shuffle=False,
        )
        if validation_data is not None
        else None
    )

    model = SegmentClassifier(hparams["model_name"], hparams["dropout"]).to(device)
    class_counts = train_data["label"].value_counts().reindex([0, 1], fill_value=0)
    class_weights = len(train_data) / (2 * class_counts.clip(lower=1))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights.to_numpy(), dtype=torch.float32, device=device
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
    )
    total_steps = hparams["num_epochs"] * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * hparams["warmup_ratio"]), total_steps
    )
    mixed_precision_dtype = mp_str_to_dtype[hparams["mixed_precision_dtype"]]
    training_results, best_state = train(
        model,
        train_loader,
        optimizer,
        scheduler,
        criterion,
        device,
        hparams["num_epochs"],
        val_loader=validation_loader,
        mixed_precision_dtype=mixed_precision_dtype,
        patience=hparams["patience"],
    )

    model.load_state_dict(best_state)
    if save:
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, fold_dir / "model.pt")
    return model, criterion, training_results


def train_lodo(df, model_name, hparams, val, model_output_dir, save=None, trial=None):
    """Train and evaluate a segment classifier with leave-one-debate-out CV."""
    if "debate_id" not in df:
        raise ValueError("LODO training requires a 'debate_id' column")
    if val and "spans" not in df:
        raise ValueError("--val requires a 'spans' column for get_validation_debate")

    debates = df["debate_id"].dropna().drop_duplicates().tolist()
    if len(debates) < 2:
        raise ValueError("LODO training requires at least two debate_id groups")
    if val and len(debates) < 3:
        raise ValueError("--val requires at least three debate_id groups")

    tokenizer = get_tokenizer(model_name)
    device = get_device()

    console.print(f"Training on {len(debates)} debates with {len(df)} segments")
    console.print(f"Using model {model_name}")
    console.print(f"Hyperparameters:")
    [console.print(f"  {key}: {value}") for key, value in hparams.items()]

    fold_results = {}
    test_predictions = {}
    test_labels = {}
    if not trial:
        progress.start()
    progress_folds = progress.add_task("Folds", total=len(debates))
    overall_training_start = time.perf_counter()
    try:
        for fold_index, test_debate in enumerate(debates):
            fold_training_start = time.perf_counter()
            set_random_seed(hparams["seed"] + fold_index)
            remaining_debates = [debate for debate in debates if debate != test_debate]
            test_data = df[df["debate_id"] == test_debate]
            validation_debate = None
            validation_data = None

            if val:
                validation_debate = get_validation_debate(df, remaining_debates)
                validation_data = df[df["debate_id"] == validation_debate]
                train_data = df[~df["debate_id"].isin([test_debate, validation_debate])]
            else:
                train_data = df[df["debate_id"] != test_debate]

            unique_columns = [
                column for column in ("id", "chunk_id", "sentence_num") if column in df
            ]
            if unique_columns:
                train_data = train_data.drop_duplicates(subset=unique_columns)
                test_data = test_data.drop_duplicates(subset=unique_columns)
                if validation_data is not None:
                    validation_data = validation_data.drop_duplicates(
                        subset=unique_columns
                    )

            console.rule(f"Fold {fold_index + 1}: {test_debate}")
            train_size = len(train_data)
            test_size = len(test_data)
            if validation_debate is not None:
                val_size = len(validation_data)
                console.print(f"Validation debate: {validation_debate}")
                console.print(
                    f"Sizes: {train_size} ({train_size/len(df)*100:.1f}%) - {val_size} ({val_size/len(df)*100:.1f}%) - {test_size} ({test_size/len(df)*100:.1f}%)"
                )
            else:
                console.print(
                    f"Sizes: {train_size} ({train_size/len(df)*100:.1f}%) - {test_size} ({test_size/len(df)*100:.1f}%)"
                )

            model, criterion, training_results = train_fold(
                train_data,
                validation_data,
                hparams,
                tokenizer,
                device,
                (
                    model_output_dir / "folds" / str(test_debate)
                    if model_output_dir
                    else None
                ),
                save=save,
            )
            test_loader = make_loader(
                test_data,
                tokenizer,
                hparams["batch_size"],
                "label",
                shuffle=False,
            )
            test_loss, test_metrics, predictions, labels = evaluate(
                model, test_loader, criterion, device
            )
            test_predictions[str(test_debate)] = predictions
            test_labels[str(test_debate)] = labels
            console.print(
                f"Test: F1={test_metrics['f1']*100:.1f} P={test_metrics['precision']*100:.1f} R={test_metrics['recall']*100:.1f}"
            )

            fold_result = {
                **training_results,
                "test_loss": test_loss,
                "test_metrics": test_metrics,
                "validation_debate": validation_debate,
                "train_size": len(train_data),
                "validation_size": (
                    0 if validation_data is None else len(validation_data)
                ),
                "test_size": len(test_data),
                "training_time_seconds": time.perf_counter() - fold_training_start,
            }
            if val:
                fold_result["best_validation_metrics"] = training_results[
                    "best_validation_metrics"
                ]
            fold_results[str(test_debate)] = fold_result

            # Report to Optuna trial if provided
            if trial is not None:
                # Report the mean macro F1 score across all completed folds to Optuna for pruning decisions
                cumulative_mean_macro_f1 = np.array(
                    [
                        mr["best_validation_metrics"]["f1"]
                        for mr in fold_results.values()
                    ]
                ).mean()
                trial.report(cumulative_mean_macro_f1, fold_index)
                console.print(
                    f"Trial report: Cumulative mean M-F1={cumulative_mean_macro_f1:.2%}"
                )

                # Check if the trial should be pruned
                if trial.should_prune():
                    trial.set_user_attr(
                        "partial_trial_data",
                        {
                            "hparams": hparams,
                            "results": fold_results,
                            "deciding_metric": float(cumulative_mean_macro_f1),
                            "pruned": True,
                            "completed_folds": fold_index + 1,
                            "pruned_on_debate": test_debate,
                        },
                    )
                    console.print(
                        f"Trial pruned at fold {fold_index + 1} for debate {test_debate}"
                    )
                    raise TrialPruned()

            progress.advance(progress_folds)
    finally:
        progress.remove_task(progress_folds)
        if not trial:
            progress.stop()

    def aggregate_metrics(metric_key):
        return {
            "mean": {
                metric: float(
                    np.mean(
                        [result[metric_key][metric] for result in fold_results.values()]
                    )
                )
                for metric in ("accuracy", "precision", "recall", "f1")
            },
            "std": {
                metric: float(
                    np.std(
                        [result[metric_key][metric] for result in fold_results.values()]
                    )
                )
                for metric in ("accuracy", "precision", "recall", "f1")
            },
        }

    aggregate_test_metrics = aggregate_metrics("test_metrics")
    aggregate_validation_metrics = None
    if val:
        aggregate_validation_metrics = aggregate_metrics("best_validation_metrics")
    results = {
        **fold_results,
        "overall": {
            "test": aggregate_test_metrics,
            "training_time_seconds": time.perf_counter() - overall_training_start,
            "best_epochs": [result["best_epoch"] for result in fold_results.values()],
            "best_epoch_median": sorted(
                result["best_epoch"] for result in fold_results.values()
            )[len(fold_results) // 2],
        },
    }
    if val:
        results["overall"]["validation"] = aggregate_validation_metrics
        console.print(
            f"Overall validation metrics: "
            f"F1={aggregate_validation_metrics['mean']['f1']*100:.1f} "
            f"P={aggregate_validation_metrics['mean']['precision']*100:.1f} "
            f"R={aggregate_validation_metrics['mean']['recall']*100:.1f}"
        )

    console.print(
        f"Overall test metrics: "
        f"F1={aggregate_test_metrics['mean']['f1']*100:.1f} "
        f"P={aggregate_test_metrics['mean']['precision']*100:.1f} "
        f"R={aggregate_test_metrics['mean']['recall']*100:.1f}"
    )

    if model_output_dir is not None:
        with (model_output_dir / "test_preds_labels.json").open("w") as file:
            json.dump(
                {"preds": test_predictions, "labels": test_labels},
                file,
                ensure_ascii=False,
                indent=4,
            )

    return results


def main():
    args = parse_args()
    if args.batch_size < 1 or args.num_epochs < 1 or args.patience < 1:
        raise ValueError(
            "--batch-size, --num-epochs, and --patience must be at least 1"
        )
    set_random_seed(args.seed)

    hparams = {
        "model_name": args.model_name,
        "num_epochs": args.num_epochs,
        "lr": args.learning_rate,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "warmup_ratio": args.warmup_ratio,
        "patience": args.patience,
        "mixed_precision_dtype": args.mixed_precision_dtype,
        "seed": args.seed,
    }

    dataframe = load_data(args.dataset_path)

    output_dir = get_model_output_dir("segment", args.model_name, args.comment)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = train_lodo(
        dataframe, args.model_name, hparams, args.val, output_dir, args.save
    )

    with (output_dir / "results.json").open("w") as file:
        json.dump(results, file, indent=2)
    console.print(f"Saved fold results to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
