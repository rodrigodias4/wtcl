import argparse
import ast
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.utils import logging
from torchcrf import CRF

from plot import plot_metric_curves, plot_train_loss_curve, plot_train_val_loss_curves
from utils import (
    MODEL_DEFAULT,
    decay_group,
    get_optimizer,
    get_validation_debate,
    label2id,
    id2label,
    label_list,
    no_decay_group,
)

logging.set_verbosity_error()
logging.disable_progress_bar()

MAX_LENGTH = 512
EARLY_STOPPING_DELTA = 0.01
PATIENCE = 3

script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

# Set random seeds for reproducibility
RANDOM_SEED = 42


def set_random_seed(seed: int) -> None:
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model_output_dir(dataset_name: str) -> Path:
    return (
        Path(os.path.dirname(os.path.abspath(__file__)))
        / "models"
        / dataset_name
        / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )


# Device setup
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        raise RuntimeError(
            "No suitable device found. Please use a machine with CUDA or MPS support."
        )
    return device


class WTCLModel(nn.Module):
    def __init__(self, model_name: str, hparams: dict = None):
        super(WTCLModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(
            model_name,
            device_map=get_device(),
        )
        if hparams:
            self.transformer.config.hidden_dropout_prob = 0.1
            self.transformer.config.attention_probs_dropout_prob = 0.1

        hidden_size = self.transformer.config.hidden_size

        self.dropout = nn.Dropout(hparams["dropout"])
        self.fc = nn.Linear(hidden_size, len(label_list), device=get_device())
        self.use_crf = hparams["use_crf"]
        if self.use_crf:
            self.crf = CRF(len(label_list), batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> dict:
        transformer_output = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )
        dropout_output = self.dropout(transformer_output.last_hidden_state)
        logits = self.fc(dropout_output)
        result = {}

        if self.use_crf:
            if labels is not None:
                # Set padding token labels to 0 for loss computation (ignored in CRF with mask)
                labels = labels.clone().detach()
                labels[labels == -100] = 0

                # Training loss is negative log-likelihood from CRF
                loss = -self.crf(
                    logits, labels, mask=attention_mask.bool(), reduction="mean"
                )
                result["loss"] = loss

            result["predictions"] = self.crf.decode(logits, mask=attention_mask.bool())
        else:
            if labels is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="mean",
                )
                result["loss"] = loss

            result["predictions"] = torch.argmax(logits, dim=-1).cpu().tolist()

        return result


def build_model(model_name: str, hparams: dict = None) -> nn.Module:
    return WTCLModel(model_name, hparams)


# Tokenizer
def get_tokenizer(model_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


def encode(
    text: torch.Tensor,
    labels: torch.Tensor,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> dict:
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    enc["labels"] = [label2id[label] for label in labels if label in label2id] + [-100] * (max_length - len(labels))
    if len(enc["labels"]) != max_length:
        raise ValueError(
            f"Warning: Labels length {len(enc['labels'])} does not match max_length {max_length}."
        )
    return enc


# Dataset definition
class WTCLDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        enc = encode(
            item["text"],
            ast.literal_eval(item["labels"]),
            self.tokenizer,
            self.max_length,
        )

        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        labels = torch.tensor(enc["labels"], dtype=torch.long)
        assert input_ids.shape[-1] == labels.shape[-1], "Input and label lengths must match max_length"

        # Set labels to -100 for padding tokens
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# --------------------------------
# Evaluation Metrics Computation
# --------------------------------


def compute_micro_f1(metrics: dict) -> float:
    tp = sum([f["precision"] * f["support"] for f in metrics])
    fp = sum([(1 - f["precision"]) * f["support"] for f in metrics])
    fn = sum([(1 - f["recall"]) * f["support"] for f in metrics])

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    micro_f1 = (2 * precision * recall) / (precision + recall + 1e-9)
    return micro_f1


def compute_metrics_token_level(preds: list, labels: list, num_labels: int = 3) -> dict:
    # Pad predictions and labels to the same length
    flat_preds = []
    flat_labels = []

    for pred, label in zip(preds, labels):
        # remove padding labels
        valid_l = [x for x in label if x != -100]

        # align lengths safely
        pred = pred[:len(valid_l)]

        flat_preds.extend(pred)
        flat_labels.extend(valid_l)

    preds = np.array(flat_preds)
    labels = np.array(flat_labels)

    metrics = {}

    # Compute per-class metrics
    for c in range(num_labels):
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        accuracy = tp / (tp + fp + fn + 1e-9)

        f1 = (2 * precision * recall) / (precision + recall + 1e-9)
        metrics[id2label[c]] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "support": int(np.sum(labels == c)),
        }

    # Compute macro and micro metrics
    metrics["macro"] = {
        "f1": np.mean([m["f1"] for m in metrics.values()]),
        "precision": np.mean([m["precision"] for m in metrics.values()]),
        "recall": np.mean([m["recall"] for m in metrics.values()]),
        "accuracy": np.mean([m["accuracy"] for m in metrics.values()]),
        "support": int(np.sum([m["support"] for m in metrics.values()])),
    }
    metrics["micro"] = {
        "f1": compute_micro_f1(list(metrics.values())),
        "precision": np.sum([m["precision"] * m["support"] for m in metrics.values()])
        / np.sum([m["support"] for m in metrics.values()]),
        "recall": np.sum([m["recall"] * m["support"] for m in metrics.values()])
        / np.sum([m["support"] for m in metrics.values()]),
        "accuracy": np.sum([m["accuracy"] * m["support"] for m in metrics.values()])
        / np.sum([m["support"] for m in metrics.values()]),
        "support": int(np.sum([m["support"] for m in metrics.values()])),
    }

    return metrics


# ---------------------------------
# Training and Evaluation Functions
# ---------------------------------


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple:
    model.eval()

    if len(dataloader) == 0:
        raise ValueError(
            "No batches were processed. Check the dataloader and input data."
        )
    all_preds = []
    all_labels = []
    total_loss = 0
    total_sequences = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_size = batch["input_ids"].size(0)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            all_preds.extend(outputs["predictions"])
            all_labels.extend(labels.cpu().tolist())
            total_loss += outputs["loss"].item() * batch_size
            total_sequences += batch_size

    return all_preds, all_labels, total_loss / total_sequences


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epochs: int,
    val_loader: DataLoader = None,
) -> dict:
    model_results = {}
    model_results["training_loss"] = []
    model_results["validation_loss"] = []
    model_results["validation_metrics"] = []
    best_macro_f1 = -1.0
    epochs_no_improve = 0
    for epoch in tqdm(range(epochs), leave=False, desc="Training Epochs"):
        model.train()
        total_training_loss = 0
        total_sequences = 0

        for batch in tqdm(train_loader, leave=False, desc="Batches"):
            optimizer.zero_grad()
            batch_size = batch["input_ids"].size(0)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs["loss"]
            loss.backward()

            optimizer.step()
            scheduler.step()

            total_sequences += batch_size
            total_training_loss += loss.item() * batch_size

        training_loss = total_training_loss / total_sequences
        model_results["training_loss"].append(training_loss)

        if val_loader is not None:
            preds, labels, val_loss = evaluate(model, val_loader, get_device())
            """ logits = torch.tensor(logits, dtype=torch.float32)
            val_loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                torch.tensor(labels, dtype=torch.long).view(-1),
                ignore_index=-100,
                reduction="mean",
            ) """
            model_results["validation_loss"].append(val_loss)
            validation_metrics = compute_metrics_token_level(preds, labels)
            model_results["validation_metrics"].append(validation_metrics)

            # Early stopping
            if validation_metrics["macro"]["f1"] > best_macro_f1 + EARLY_STOPPING_DELTA:
                best_macro_f1 = validation_metrics["macro"]["f1"]
                epochs_no_improve = 0
                validation_metrics["best_epoch"] = epoch
                validation_metrics["best_macro_f1"] = best_macro_f1
            else:
                epochs_no_improve += 1
            validation_metrics["epochs_no_improve"] = epochs_no_improve
            tqdm.write(
                f"░░ Epoch {epoch}: "
                f"TL={training_loss:.4f}, "
                f"VL={val_loss:.4f}, "
                f"M-F1={validation_metrics['macro']['f1']:.4f}, "
                f"m-F1={validation_metrics['micro']['f1']:.4f}, "
                f"A={validation_metrics['macro']['accuracy']:.4f}, "
                f"P={validation_metrics['macro']['precision']:.4f}, "
                f"R={validation_metrics['macro']['recall']:.4f}\t"
                f"{'↓' * validation_metrics['epochs_no_improve']}"
            )
            if epochs_no_improve >= PATIENCE:
                tqdm.write(
                    f"Early stopping at epoch {epoch} due to no improvement in validation f1."
                )
                break
        else:
            tqdm.write(f"Epoch {epoch}: TL={training_loss:.4f}")
        gc.collect()
        torch.cuda.empty_cache()
    return model_results


def train_lodo(
    df: pd.DataFrame, model_name: str, hparams: dict, val: bool, model_output_dir: Path
) -> dict:
    """
    Leave-one-debate-out training function.
    For each debate, we train a model on all other debates and evaluate on the left-out debate,
    and return the results and trained models for each left-out debate.
    """

    all_debates = df["debate_id"].unique()
    results = {}
    tokenizer = get_tokenizer(model_name)
    val_loader = None

    print(f"Training with model '{model_name}'\nHyperparameters:")
    [print(f"‣ {k}: {v}") for k, v in hparams.items()]
    print(f"Leave-one-debate-out training on {len(all_debates)} debates.")
    print(f"Validation enabled: {val}")

    all_preds = []
    all_labels = []

    # Leave-one-debate-out
    for test_debate in tqdm(all_debates, desc="Leave-One-Debate-Out Folds", leave=True):
        tqdm.write(f"Training model with debate '{test_debate}' left out for testing.")
        remaining_debates = [d for d in all_debates if d != test_debate]
        if val:
            val_debate = get_validation_debate(df, remaining_debates)
            val_data = df[df["debate_id"] == val_debate]
            train_data = df[~df["debate_id"].isin([test_debate, val_debate])]
            assert (
                not train_data["debate_id"].isin([test_debate, val_debate]).any()
            ), "Training debates should not include test or validation debate"
            tqdm.write(
                f"░░ Validating on debate: {val_debate} | Train size: {len(train_data)} | Val size: {len(val_data)}"
            )
        else:
            train_data = df[df["debate_id"] != test_debate]
            assert not (
                train_data["debate_id"] == test_debate
            ).any(), "Training debates should not include test debate"
        test_data = df[df["debate_id"] == test_debate]

        # Prepare datasets and dataloaders
        train_dataset = WTCLDataset(
            train_data.to_dict("records"), tokenizer, MAX_LENGTH
        )
        test_dataset = WTCLDataset(test_data.to_dict("records"), tokenizer, MAX_LENGTH)

        train_loader = DataLoader(
            train_dataset, batch_size=hparams["batch_size"], shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=hparams["batch_size"], shuffle=False
        )
        if val:
            val_dataset = WTCLDataset(
                val_data.to_dict("records"), tokenizer, MAX_LENGTH
            )
            val_loader = DataLoader(
                val_dataset, batch_size=hparams["batch_size"], shuffle=False
            )

        # Create new model instance for each CV fold
        model = build_model(
            model_name, {"dropout": hparams["dropout"], "use_crf": hparams["use_crf"]}
        )

        # Set up optimizer and scheduler
        optimizer = get_optimizer(model, hparams)

        total_steps = len(train_loader) * hparams["num_epochs"]

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(hparams["warmup_ratio"] * total_steps),
            num_training_steps=total_steps,
        )

        # Train the model
        model_results = train(
            model,
            train_loader,
            optimizer,
            scheduler,
            get_device(),
            hparams["num_epochs"],
            val_loader,
        )

        # Save the model for this debate
        with (model_output_dir / f"{test_debate}.pt").open("wb") as f:
            torch.save(model.state_dict(), f)

        # Evaluate the model on the test debate
        preds, labels, _ = evaluate(model, test_loader, get_device())
        all_preds.extend(np.concatenate(preds).tolist())
        all_labels.extend(np.concatenate(labels).tolist())
        test_metrics = compute_metrics_token_level(preds, labels)
        model_results["test_metrics"] = test_metrics
        tqdm.write(
            f"Debate '{test_debate}' - Macro F1: {test_metrics['macro']['f1']:.4f}"
        )
        results[test_debate] = model_results

        del (
            model,
            train_loader,
            test_loader,
            train_dataset,
            test_dataset,
            preds,
            labels,
        )  # Free up memory
        gc.collect()
        torch.cuda.empty_cache()

    with (model_output_dir / f"all_preds_labels.json").open("w") as f:
        json.dump(
            {"preds": all_preds, "labels": all_labels}, f, ensure_ascii=False, indent=4
        )

    results["overall"] = {}
    results["overall"]["macro"] = {}
    results["overall"]["micro"] = {}
    for metric in ["f1", "precision", "recall"]:
        results["overall"]["macro"][metric] = np.mean(
            [results[debate]["test_metrics"]["macro"][metric] for debate in all_debates]
        )
        results["overall"]["micro"][metric] = np.mean(
            [results[debate]["test_metrics"]["micro"][metric] for debate in all_debates]
        )
    tqdm.write(
        f"Overall Macro F1 across all debates: {results['overall']['macro']['f1']:.4f}"
    )

    return results


# ---------------------------------
# Results Processing and Plotting
# ---------------------------------


def process_results(results: dict, output_dir: Path) -> None:
    # Save results to JSON
    with (output_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Print overall results
    print("\nPer-left-out-debate results:")
    for debate, metrics in results.items():
        if debate in ["overall", "hyperparameters"]:
            continue
        print(f"Debate '{debate}': M-F1 = {metrics['test_metrics']['macro']['f1']:.4f}")

    if all(
        [
            results[debate].get("validation_metrics") is not None
            for debate in results
            if debate not in ["overall", "hyperparameters"]
        ]
    ):
        # Plot validation metrics if validation was performed
        print("\nPlotting training and validation loss curves...")
        plot_train_val_loss_curves(results, output_dir)
        print("\nPlotting validation metrics curves...")
        plot_metric_curves(results, output_dir)
    else:
        # Plot training loss curve
        print("\nPlotting training loss curve...")
        plot_train_loss_curve(results, output_dir)


# -------------------------------
# Argument Parsing and Main Function
# -------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("input_file", type=str, help="Path to input CSV file")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wtcl",
        help="Name of the dataset (for saving models)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate for the model."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate scheduler.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_DEFAULT,
        help="Name of the transformer model to use.",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="Whether to perform validation during training (enables early stopping).",
    )
    parser.add_argument(
        "--no_crf",
        action="store_false",
        help="Whether to disable the CRF layer on top of the transformer model.",
    )
    return parser.parse_args()


def main():
    set_random_seed(RANDOM_SEED)
    torch.cuda.empty_cache()
    args = parse_args()

    df = pd.read_csv(args.input_file)
    print(f"Loaded data with {len(df)} rows from {args.input_file}.")

    model_output_dir = get_model_output_dir(args.dataset_name)
    model_output_dir.mkdir(exist_ok=True, parents=True)

    hparams = {
        "dropout": args.dropout,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "lr": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "use_crf": args.no_crf,
    }
    results = train_lodo(df, args.model_name, hparams, args.val, model_output_dir)
    results["hyperparameters"] = hparams

    process_results(results, model_output_dir)


if __name__ == "__main__":
    main()
