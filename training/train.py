import argparse
import ast
from datetime import datetime
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

from plot import plot_train_loss_curve
from utils import MODEL_DEFAULT, get_validation_debate, label2id, id2label, label_list

logging.set_verbosity_error()
logging.disable_progress_bar()

MAX_LENGTH = 512
EARLY_STOPPING_DELTA = 0.02
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
        / datetime.now().strftime("%Y-%m-%d-%H-%M")
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

        return result


def build_model(model_name: str, hparams: dict = None) -> nn.Module:
    return WTCLModel(model_name, hparams)


# Tokenizer
def get_tokenizer(model_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


def encode(text: torch.Tensor, labels: torch.Tensor, tokenizer: AutoTokenizer) -> dict:
    # Pad labels to match the length of input_ids
    if len(labels) < MAX_LENGTH:
        labels += ["O"] * (MAX_LENGTH - len(labels))
    elif len(labels) > MAX_LENGTH:
        raise ValueError(
            f"Warning: Labels length {len(labels)} exceeds MAX_LENGTH {MAX_LENGTH}."
        )
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    enc["labels"] = [label2id[label] for label in labels]
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

        enc = encode(item["text"], ast.literal_eval(item["labels"]), self.tokenizer)

        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        labels = torch.tensor(enc["labels"], dtype=torch.long)

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
    preds = [p + [-100] * (len(labels[0]) - len(p)) for p in preds]
    preds = np.array(preds)
    labels = np.array(labels)

    # flatten (batch, seq) → (N,)
    preds = preds.flatten()
    labels = labels.flatten()

    # remove ignored tokens
    mask = labels != -100
    preds = preds[mask]
    labels = labels[mask]

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
    model.to(device)

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
    best_val_loss = float("inf")
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
            tqdm.write(
                f"Epoch {epoch}: "
                f"TL={training_loss:.4f}, "
                f"VL={val_loss:.4f}, "
                f"M-F1={validation_metrics['macro']['f1']:.4f}, "
                f"m-F1={validation_metrics['micro']['f1']:.4f}, "
                f"A={validation_metrics['macro']['accuracy']:.4f}, "
                f"P={validation_metrics['macro']['precision']:.4f}, "
                f"R={validation_metrics['macro']['recall']:.4f}"
            )

            # Early stopping
            if (val_loss < best_val_loss - EARLY_STOPPING_DELTA):
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:
                tqdm.write(
                    f"\tEarly stopping at epoch {epoch} due to no improvement in validation loss."
                )
                break
        else:
            tqdm.write(f"Epoch {epoch}: TL={training_loss:.4f}")
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

    # Leave-one-debate-out
    for debate in tqdm(all_debates, desc="Leave-One-Debate-Out Folds", leave=True):
        remaining_debates = [d for d in all_debates if d != debate]
        if val:
            val_debate = get_validation_debate(df, remaining_debates)
            val_data = df[df["debate_id"] == val_debate]
            train_data = train_data = df[df["debate_id"] not in [debate, val_debate]]
        else:
            train_data = df[df["debate_id"] != debate]
        test_data = df[df["debate_id"] == debate]

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
        model = build_model(model_name, {"dropout": hparams["dropout"]})

        # Set up optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

        total_steps = len(train_loader) * hparams["num_epochs"]

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(hparams["warmup_ratio"] * total_steps),
            num_training_steps=total_steps,
        )

        # Train the model
        tqdm.write(f"Training model with debate '{debate}' left out for testing.")
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
        with (model_output_dir / f"{debate}").open("wb") as f:
            torch.save(model.state_dict(), f)

        # Evaluate the model on the test debate
        preds, labels, _ = evaluate(model, test_loader, get_device())
        test_metrics = compute_metrics_token_level(preds, labels)
        model_results["test_metrics"] = test_metrics
        tqdm.write(
            f"Debate '{debate}' - Macro F1: {test_metrics['macro']['f1']:.4f}"
        )
        results[debate] = model_results
        del (
            model,
            train_loader,
            test_loader,
            train_dataset,
            test_dataset,
        )  # Free up memory

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
    # Print overall results
    print("\nOverall Results:")
    for debate, metrics in results.items():
        print(f"Debate '{debate}': Macro F1 = {metrics['test_macro_f1']:.4f}")

    print(f"Overall Macro F1: {results['overall']['macro']['f1']:.4f}")

    # Save results to JSON
    with (output_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=4)

    plot_train_loss_curve(results, output_dir / "training_loss_curve.png")


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
        default=0.001,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate scheduler.",
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
    return parser.parse_args()


def main():
    set_random_seed(RANDOM_SEED)
    args = parse_args()
    print(
        f"Training for {args.num_epochs} epochs with batch size {args.batch_size} and learning rate {args.learning_rate}."
    )

    df = pd.read_csv(args.input_file)
    print(f"Loaded data with {len(df)} rows from {args.input_file}.")

    model_output_dir = get_model_output_dir(args.dataset_name)
    model_output_dir.mkdir(exist_ok=True, parents=True)

    results = train_lodo(df, args.model_name, vars(args), args.val, model_output_dir)

    process_results(results, model_output_dir)


if __name__ == "__main__":
    main()
