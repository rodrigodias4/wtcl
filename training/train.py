import argparse
import ast
from collections import defaultdict
from datetime import datetime
import gc
import json
from math import ceil
import os
from pathlib import Path
import random
import signal

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, jaccard_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast, GradScaler

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from transformers.utils import logging
from torchcrf import CRF
from optuna.trial import Trial
from optuna.exceptions import TrialPruned

from plot import (
    plot_validation_metric_curves,
    plot_train_loss_curve,
    plot_train_val_loss_curves,
)
from plot_cm import compute_metrics_span_level, plot_confusion_matrix
from utils import (
    EMISSION_BIAS_B,
    EMISSION_BIAS_I,
    LEARNING_RATE_CRF_MULTIPLIER,
    LEARNING_RATE_FC_MULTIPLIER,
    MODEL_DEFAULT,
    get_optimizer,
    get_validation_debate,
    handle_interrupt,
    label2id,
    id2label,
    label_list,
    console,
    progress,
)

logging.set_verbosity_error()
logging.disable_progress_bar()
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

MAX_LENGTH = 512
EARLY_STOPPING_DELTA = 0.0045  # Minimum improvement in validation macro F1 to reset early stopping counter (0.45%)
PATIENCE = 3  # Number of epochs to wait for improvement before early stopping

RANDOM_SEED = 42

B_ID = label2id["B"]
I_ID = label2id["I"]
O_ID = label2id["O"]

B_ID_str = str(B_ID)
I_ID_str = str(I_ID)
O_ID_str = str(O_ID)

mp_str_to_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "none": None,
}

signal.signal(signal.SIGINT, handle_interrupt)


# -------------------------------
# Helper Functions
# -------------------------------


def set_random_seed(seed: int) -> None:
    """
    Helper function to seed experiment for reproducibility.
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
    torch.use_deterministic_algorithms(True)
    set_seed(seed)


def get_model_output_dir(dataset_name: str, model_name: str, comment: str = "") -> Path:
    return (
        Path(os.path.dirname(os.path.abspath(__file__)))
        / "models"
        / dataset_name
        / model_name.split("/")[-1]
        / (
            datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            + (f" {comment}" if comment else "")
        )
    )


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


DEVICE = get_device()

# --------------------------------
# Model Definition
# --------------------------------


class WTCLModel(nn.Module):
    def __init__(self, model_name: str, hparams: dict = None):
        super(WTCLModel, self).__init__()
        self.hparams = hparams

        config = AutoConfig.from_pretrained(model_name)
        transformer_dropout = 0.1
        dropout_attributes = [
            "hidden_dropout_prob",
            "hidden_dropout",
            "attention_probs_dropout_prob",
            "attention_dropout",
            "dropout",
            "resid_pdrop",
        ]
        for attr in dropout_attributes:
            if hasattr(config, attr):
                setattr(config, attr, transformer_dropout)

        self.transformer = AutoModel.from_pretrained(
            model_name,
            config=config,
            dtype=torch.float32,
        )

        # Enable gradient checkpointing to save memory
        if hparams["gradient_checkpointing"]:
            self.transformer.gradient_checkpointing_enable()
            if hasattr(self.transformer, "enable_input_require_grads"):
                self.transformer.enable_input_require_grads()

        # Freeze the first N layers of the transformer if specified in hyperparameters
        for name, param in self.transformer.named_parameters():
            for i in range(hparams["freeze"]):
                if f"encoder.layer.{i}" in name:
                    param.requires_grad = False

        # Set emission bias for CRF if specified in hyperparameters
        emission_bias = [0.0 for _ in range(len(label_list))]
        if hparams["emission_bias"]:
            emission_bias[B_ID] = EMISSION_BIAS_B
            emission_bias[I_ID] = EMISSION_BIAS_I
            self.emission_bias = torch.tensor(
                emission_bias, dtype=torch.float32, device=DEVICE
            )

        hidden_size = self.transformer.config.hidden_size

        self.dropout = nn.Dropout(hparams["dropout"])
        self.fc = nn.Linear(hidden_size, len(label_list), device=DEVICE)
        self.use_crf = hparams["use_crf"]
        if self.use_crf:
            self.crf = CRF(len(label_list), batch_first=True)

        # Set CRF priors if specified in hyperparameters
        if hparams["crf_priors"] and self.crf is not None:
            with torch.no_grad():
                self.crf.transitions[O_ID, B_ID] = 1.0
                self.crf.transitions[O_ID, I_ID] = -10000.0
                self.crf.transitions[B_ID, O_ID] = -1.0
                self.crf.transitions[B_ID, B_ID] = -1.0
                self.crf.transitions[B_ID, I_ID] = 1.0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        crf_mask: torch.Tensor = None,
    ) -> dict:
        # Get transformer output
        transformer_output = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )
        # Apply dropout
        dropout_output = self.dropout(transformer_output.last_hidden_state)
        del transformer_output
        # Compute logits
        logits = self.fc(dropout_output)
        del dropout_output

        # Add emission bias to logits for CRF
        if self.hparams["emission_bias"]:
            logits = logits + self.emission_bias

        if torch.isnan(logits).any():
            console.print(f"[WARNING] NaN detected in logits!")

        result = {}

        if self.use_crf:
            mask = attention_mask.bool() & crf_mask.bool()

            # Compress sequences to word-level
            logits_comp = pad_sequence(
                [logit_seq[mask_seq] for logit_seq, mask_seq in zip(logits, mask)],
                batch_first=True,
            )

            # Explicitly convert logits to float32 for CRF computations to avoid potential issues with mixed precision
            logits_comp_fp32 = logits_comp.float()
            del logits_comp

            lengths = mask.sum(dim=1)
            mask_comp = torch.arange(lengths.max(), device=mask.device).expand(
                len(lengths), -1
            ) < lengths.unsqueeze(1)
            del lengths

            if labels is not None:
                labels_comp = pad_sequence(
                    [label_seq[mask_seq] for label_seq, mask_seq in zip(labels, mask)],
                    batch_first=True,
                    padding_value=0,  # ignored by the CRF due to mask_comp
                )

                # Replace any remaining ignore indices (shouldn't normally exist)
                labels_comp = labels_comp.masked_fill(labels_comp == -100, 0)

                result["loss"] = -self.crf(
                    logits_comp_fp32,
                    labels_comp,
                    mask=mask_comp,
                    reduction="mean",
                )

            if not self.training:
                result["predictions"] = self.crf.decode(
                    logits_comp_fp32,
                    mask=mask_comp,
                )

            del logits_comp_fp32, mask_comp, labels_comp, mask
        else:
            mask = attention_mask.bool() & crf_mask.bool()
            if labels is not None:
                # Compute cross-entropy loss for non-CRF case
                loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="mean",
                )
                result["loss"] = loss

            if not self.training:
                # Get predictions by taking the argmax of logits for non-CRF case
                predictions = torch.argmax(logits, dim=-1).cpu().tolist()

                # Filter predictions to only include non-padding and first-subword tokens
                if isinstance(predictions, list) and isinstance(predictions[0], list):
                    predictions = [
                        [pred for pred, m in zip(pred_seq, mask_seq) if m]
                        for pred_seq, mask_seq in zip(predictions, mask.cpu().tolist())
                    ]

                result["predictions"] = predictions
        del logits
        return result


def build_model(model_name: str, hparams: dict = None) -> nn.Module:
    # Currently, we only have one model class, but this function allows for easy extension in the future.
    return WTCLModel(model_name, hparams)


# Tokenizer
def get_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Get the tokenizer for the specified model.
    @param model_name: Name of the transformer model to use.
    @return: AutoTokenizer instance for the specified model.
    """

    return AutoTokenizer.from_pretrained(
        model_name,
        add_prefix_space=(
            model_name.split("/")[-1] in ["roberta-base", "roberta-large"]
        ),
    )


def encode(
    text: str,
    spans: list[dict],
    tokenizer: AutoTokenizer,
    max_length: int = MAX_LENGTH,
) -> dict:
    """
    Encode text and labels for the model.

    @param text: Input text to encode.
    @param spans: List of dictionaries representing the spans to tag.
    @param tokenizer: Tokenizer to use for encoding.
    @param max_length: Maximum sequence length for padding/truncation.
    @return: Dictionary containing encoded input_ids, attention_mask, and labels.
    """
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    labels = [label2id["O"]] * len(enc["input_ids"])

    def token_overlaps_span(tok_start, tok_end, span_start, span_end):
        return tok_start < span_end and tok_end > span_start

    # Assign BIO labels to tokens based on the provided spans
    for span in spans:
        span_start = span["start"]
        span_end = span["end"]

        token_indices = []

        for i, (tok_start, tok_end) in enumerate(enc["offset_mapping"]):
            if token_overlaps_span(tok_start, tok_end, span_start, span_end):
                token_indices.append(i)

        if not token_indices:
            continue

        # BIO tagging
        labels[token_indices[0]] = label2id["B"]
        for idx in token_indices[1:]:
            labels[idx] = label2id["I"]

    enc["labels"] = labels
    assert len(enc["input_ids"]) == len(
        enc["labels"]
    ), "Input IDs and labels must be the same length"

    # Create a CRF mask to filter non-first subword tokens
    word_ids = enc.word_ids()
    crf_mask = []
    previous_word = None

    for word_id, label in zip(word_ids, labels):
        if word_id is None:
            # Special tokens / padding
            crf_mask.append(False)
        elif label == B_ID:
            # Never drop an entity start - had to add this because the CRF mask was dropping B labels in some cases (e.g. weird transcripts where a hyphen was used as a punctuation mark without a space, causing the tokenizer to split the word into subwords)
            crf_mask.append(True)
            previous_word = word_id
        elif word_id != previous_word:
            # First subword of a word
            crf_mask.append(True)
            previous_word = word_id
        else:
            # Continuation subword
            crf_mask.append(False)

    assert len(crf_mask) == len(
        enc["input_ids"]
    ), "CRF mask and input IDs must be the same length"
    enc["crf_mask"] = crf_mask

    return enc


# --------------------------------
# Dataset Definition
# --------------------------------


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
            ast.literal_eval(item["spans"]),
            self.tokenizer,
            self.max_length,
        )

        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        labels = torch.tensor(enc["labels"], dtype=torch.long)
        crf_mask = torch.tensor(enc["crf_mask"], dtype=torch.bool)

        if len(enc["input_ids"]) != len(enc["labels"]):
            # Convert input IDs back to tokens to see what the tokenizer actually did
            actual_tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"])
            raw_labels = ast.literal_eval(item["labels"])

            raise ValueError(
                f"\nShape Mismatch at index {idx}!"
                f"\nTokenizer produced {len(enc['input_ids'])} tokens: {actual_tokens}"
                f"\nDataset has {len(enc['labels'])} labels: {raw_labels}"
                f"\nRaw Text: '{item['text']}'"
            )

        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "crf_mask": crf_mask,
        }


# --------------------------------
# Evaluation Metrics Computation
# --------------------------------


def compute_metrics_token_level(preds: list, labels: list, num_labels: int = 3) -> dict:
    """
    Compute token-level metrics (F1, precision, recall) for each class and overall.

    @param preds: List of lists of predicted labels for each token.
    @param labels: List of lists of true labels for each token.
    @param num_labels: Number of unique labels/classes.
    @return: Dictionary containing metrics for each class and macro average.
    """

    flat_preds = []
    flat_labels = []

    for pred, label in zip(preds, labels):
        assert len(pred) == len(
            label
        ), "Predictions and labels must be the same length after removing padding"

        flat_preds.extend(pred)
        flat_labels.extend(label)

    preds = np.array(flat_preds)
    labels = np.array(flat_labels)

    cr = classification_report(
        labels,
        preds,
        labels=list(range(num_labels)),
        output_dict=True,
        zero_division=0,
    )

    return {
        "macro": {
            "f1": cr["macro avg"]["f1-score"],
            "precision": cr["macro avg"]["precision"],
            "recall": cr["macro avg"]["recall"],
            "jaccard": jaccard_score(labels, preds, average="macro"),
        },
        "O": {
            "f1": cr[O_ID_str]["f1-score"],
            "precision": cr[O_ID_str]["precision"],
            "recall": cr[O_ID_str]["recall"],
            "support": cr[O_ID_str]["support"],
        },
        "B": {
            "f1": cr[B_ID_str]["f1-score"],
            "precision": cr[B_ID_str]["precision"],
            "recall": cr[B_ID_str]["recall"],
            "support": cr[B_ID_str]["support"],
        },
        "I": {
            "f1": cr[I_ID_str]["f1-score"],
            "precision": cr[I_ID_str]["precision"],
            "recall": cr[I_ID_str]["recall"],
            "support": cr[I_ID_str]["support"],
        },
    }


# ---------------------------------
# Training and Evaluation Functions
# ---------------------------------


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple:
    """
    Evaluate the model on a given dataloader and return predictions, labels, and loss.

    @param model: The model to evaluate.
    @param dataloader: DataLoader for the evaluation data.
    @param device: Device to run the evaluation on (CPU/GPU).
    @return: Tuple containing predictions, labels, and average loss.
    """
    model.eval()

    if len(dataloader) == 0:
        raise ValueError(
            "No batches were processed. Check the dataloader and input data."
        )
    all_preds = []
    all_labels = []
    total_loss = 0
    total_sequences = 0

    with torch.inference_mode():
        for batch in dataloader:
            batch_size = batch["input_ids"].size(0)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            crf_mask = batch["crf_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                crf_mask=crf_mask,
            )

            all_preds.extend(outputs["predictions"])

            labels_masked = [
                labels_i[mask_i].cpu().tolist()
                for labels_i, mask_i in zip(labels, crf_mask)
            ]
            all_labels.extend(labels_masked)

            total_loss += outputs["loss"].item() * batch_size
            total_sequences += batch_size

            del outputs, input_ids, attention_mask, labels, crf_mask

    return all_preds, all_labels, total_loss / total_sequences


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epochs: int,
    val_loader: DataLoader = None,
    mixed_precision_dtype: torch.dtype = None,
) -> tuple[dict, dict]:
    """
    Train the model with optional validation and early stopping.
    Returns the training results and the best model state.

    @param model: The model to train.
    @param train_loader: DataLoader for the training data.
    @param optimizer: Optimizer for training.
    @param scheduler: Learning rate scheduler.
    @param device: Device to run the training on (CPU/GPU).
    @param epochs: Number of training epochs.
    @param val_loader: Optional DataLoader for validation data.
    @return: Tuple containing training results and the best model state.
    """
    model_results = {}
    model_results["training_loss"] = []
    model_results["validation_loss"] = []
    model_results["validation_metrics"] = []
    best_macro_f1 = -1.0
    epochs_no_improve = 0
    best_epoch = 0
    best_model_state = None
    validation_preds = []
    validation_labels = []
    scaler = GradScaler(enabled=(mixed_precision_dtype == torch.float16))

    progress_task_epochs = progress.add_task("Training Epochs", total=epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        total_training_loss = 0
        total_sequences = 0

        progress_task_batches = progress.add_task(f"Batches", total=len(train_loader))
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch_size = batch["input_ids"].size(0)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            crf_mask = batch["crf_mask"].to(device)

            if mixed_precision_dtype is None:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    crf_mask=crf_mask,
                )
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()
                scheduler.step()
            elif mixed_precision_dtype == torch.float16:
                with autocast(
                    device_type="cuda",
                    dtype=mixed_precision_dtype,
                    cache_enabled=False,
                ):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        crf_mask=crf_mask,
                    )
                    loss = outputs["loss"]

                scaler.scale(loss).backward()

                scaler.step(optimizer)

                scale = scaler.get_scale()
                scaler.update()
                if not scale > scaler.get_scale():
                    scheduler.step()
            elif mixed_precision_dtype == torch.bfloat16:
                with autocast(
                    device_type="cuda",
                    dtype=mixed_precision_dtype,
                    cache_enabled=False,
                ):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        crf_mask=crf_mask,
                    )
                    loss = outputs["loss"]

                loss.backward()
                optimizer.step()
                scheduler.step()
            else:
                raise ValueError(
                    f"Unsupported mixed precision dtype: {mixed_precision_dtype}"
                )

            total_sequences += batch_size
            total_training_loss += loss.item() * batch_size
            del loss, outputs, input_ids, attention_mask, labels

            progress.advance(progress_task_batches)
        progress.remove_task(progress_task_batches)
        training_loss = total_training_loss / total_sequences
        model_results["training_loss"].append(training_loss)

        if val_loader is not None:
            preds, labels, val_loss = evaluate(model, val_loader, DEVICE)
            model_results["validation_loss"].append(val_loss)
            validation_metrics = compute_metrics_token_level(preds, labels)
            validation_metrics["span"] = compute_metrics_span_level(preds, labels)
            model_results["validation_metrics"].append(validation_metrics)

            # Early stopping
            if validation_metrics["macro"]["f1"] > best_macro_f1 + EARLY_STOPPING_DELTA:
                # Save current best model state, epoch and macro F1 score
                best_macro_f1 = validation_metrics["macro"]["f1"]
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_epoch = epoch
                model_results["best_epoch"] = best_epoch

                # Save validation predictions and labels for confusion matrix plotting
                validation_preds = preds
                validation_labels = labels

                # Reset early stopping counter
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            console.print(
                f"{('[magenta]░░ [/magenta]') if epoch == best_epoch else '░░ '}"
                f"Epoch {(str(epoch) + ":"):<3} "
                f"TL={training_loss:<6.2f} "
                f"VL={val_loss:<6.2f} "
                f"F1={validation_metrics['macro']['f1']:<6.2%} "
                f"P={validation_metrics['macro']['precision']:<5.1%} "
                f"R={validation_metrics['macro']['recall']:<5.1%} "
                f"B-F1={validation_metrics['B']['f1']:<5.1%} "
                f"B-P={validation_metrics['B']['precision']:<5.1%} "
                f"B-R={validation_metrics['B']['recall']:<5.1%} "
                f"I-F1={validation_metrics['I']['f1']:<5.1%} "
                f"I-P={validation_metrics['I']['precision']:<5.1%} "
                f"I-R={validation_metrics['I']['recall']:<5.1%} "
                f"O-F1={validation_metrics['O']['f1']:<5.1%} "
                f"S-F1={validation_metrics['span']['f1']:<5.1%} "
                f"J={validation_metrics['macro']['jaccard']:<5.1%}"
            )
            if epochs_no_improve >= model.hparams["patience"]:
                console.print(
                    f"Early stopping at epoch {best_epoch} due to no improvement in validation f1."
                )
                break
        else:
            console.print(f"░░ Epoch {epoch}: TL={training_loss:.2f}")
        progress.advance(progress_task_epochs)
        progress.refresh()
        gc.collect()

    # If no validation loader is provided, save the last model state as the best model state
    if val_loader is None:
        best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        best_epoch = epoch
        model_results["best_epoch"] = best_epoch
    progress.remove_task(progress_task_epochs)
    return model_results, best_model_state, validation_preds, validation_labels


def train_lodo(
    df: pd.DataFrame,
    model_name: str,
    hparams: dict,
    val: bool,
    model_output_dir: Path,
    save: bool = False,
    trial: Trial = None,
) -> tuple[dict, list, list, list, list]:
    """
    Leave-one-debate-out training function.
    For each debate, we train a model on all other debates and evaluate on the left-out debate,
    and return the results and trained models for each left-out debate.

    @param df: DataFrame containing the dataset with 'debate_id', 'text', and 'labels'.
    @param model_name: Name of the transformer model to use.
    @param hparams: Dictionary of hyperparameters for training.
    @param val: Boolean indicating whether to perform validation during training.
    @param model_output_dir: Directory to save the trained models and results.
    @param save: Boolean indicating whether to save the trained models to disk.
    @param trial: Optional Optuna trial object for hyperparameter optimization.
    @return: Tuple containing the results dictionary and lists of test predictions, test labels, validation predictions, and validation labels.
    """
    all_debates = df["debate_id"].unique()
    results = {}
    tokenizer = get_tokenizer(model_name)
    val_loader = None

    console.print(f"Training with model '{model_name}'\nHyperparameters:")
    [console.print(f"‣ {k}: {v}") for k, v in hparams.items()]
    console.print(
        f"Tokenizer: {tokenizer.__class__.__name__} | {tokenizer._tokenizer.model.__class__.__name__} | {tokenizer.vocab_size // 1000}K vocab size | is_fast={tokenizer.is_fast} | add_prefix_space={tokenizer.add_prefix_space}"
    )
    console.print(f"Leave-one-debate-out training on {len(all_debates)} debates.")
    console.print(f"Validation enabled: {val}")

    all_test_preds = {debate: [] for debate in all_debates}
    all_test_labels = {debate: [] for debate in all_debates}
    all_validation_preds = {debate: [] for debate in all_debates} if val else {}
    all_validation_labels = {debate: [] for debate in all_debates} if val else {}
    mixed_precision_dtype = mp_str_to_dtype.get(hparams["mixed_precision_dtype"])

    if model_output_dir is not None:
        if save:
            os.makedirs(model_output_dir / "folds", exist_ok=True)
        with (model_output_dir / "hyperparameters.json").open("w") as f:
            json.dump(hparams, f, indent=4, ensure_ascii=False)

    if not trial:
        progress.start()
    progress_folds = progress.add_task(
        "Leave-One-Debate-Out Folds", total=len(all_debates)
    )
    # Leave-one-debate-out
    for i, test_debate in enumerate(all_debates):
        # Reset RNG state per fold so prior trials/folds do not leak into this run.
        set_random_seed(hparams["seed"] + i)

        console.rule(f"Fold {i + 1} for test debate: {test_debate}")
        remaining_debates = [d for d in all_debates if d != test_debate]
        # Create test split dataset and dataloader
        test_data = df[df["debate_id"] == test_debate].drop_duplicates(
            subset=["id", "chunk_id"]
        )  # Ensure unique sequences in test set (no oversampling)

        # Split the data into training, validation, and test sets
        if val:
            val_debate = get_validation_debate(df, remaining_debates)
            val_data = df[df["debate_id"] == val_debate].drop_duplicates(
                subset=["id", "chunk_id"]
            )  # Ensure unique sequences in validation set (no oversampling)
            train_data = df[~df["debate_id"].isin([test_debate, val_debate])]

            total_size = len(train_data) + len(val_data) + len(test_data)

            console.print(f"Validating on debate: {val_debate}")
            console.print(
                f"Split sizes: {len(train_data)} - {len(val_data)} - {len(test_data)} // {len(train_data)/total_size:.1%} - {len(val_data)/total_size:.1%} - {len(test_data)/total_size:.1%}"
            )
        else:
            train_data = df[df["debate_id"] != test_debate]
            total_size = len(train_data) + len(test_data)
            console.print(
                f"Split sizes: {len(train_data)} - {len(test_data)} // {len(train_data)/total_size:.1%} - {len(test_data)/total_size:.1%}"
            )

        # Prepare datasets and dataloaders
        train_dataset = WTCLDataset(
            train_data.to_dict("records"), tokenizer, MAX_LENGTH
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=hparams["batch_size"],
        )

        if val:
            val_dataset = WTCLDataset(
                val_data.to_dict("records"), tokenizer, MAX_LENGTH
            )
            val_loader = DataLoader(
                val_dataset, batch_size=hparams["batch_size"], shuffle=False
            )

        # Create new model instance for each CV fold
        model = build_model(model_name, hparams).to(DEVICE)

        # Set up optimizer and scheduler
        optimizer = get_optimizer(model, hparams)

        total_steps = len(train_loader) * hparams["num_epochs"]

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(hparams["warmup_ratio"] * total_steps),
            num_training_steps=total_steps,
        )

        # Train the model
        model_results, best_model_state, validation_preds, validation_labels = train(
            model,
            train_loader,
            optimizer,
            scheduler,
            DEVICE,
            hparams["num_epochs"],
            val_loader=val_loader,
            mixed_precision_dtype=mixed_precision_dtype,
        )

        if hparams["crf_priors"]:
            console.print("Learned CRF transition parameters (with priors):")
            console.print(model.crf.transitions.clone().detach().cpu().numpy())

        # Save the best model state for this debate
        if save and best_model_state is not None and model_output_dir is not None:
            with (model_output_dir / "folds" / f"{test_debate}.pt").open("wb") as f:
                torch.save(best_model_state, f)

        test_dataset = WTCLDataset(test_data.to_dict("records"), tokenizer, MAX_LENGTH)
        test_loader = DataLoader(
            test_dataset, batch_size=hparams["batch_size"], shuffle=False
        )

        # Evaluate the best model state on the test debate
        model.load_state_dict(best_model_state)
        preds, labels, _ = evaluate(model, test_loader, DEVICE)
        all_test_preds[test_debate].extend(preds)
        all_test_labels[test_debate].extend(labels)

        # Compute token-level metrics for the test split
        test_metrics = compute_metrics_token_level(preds, labels)
        model_results["test_metrics"] = test_metrics
        results[test_debate] = model_results

        if val:
            # Store the best validation metrics for this debate in the results dictionary
            model_results["best_validation_metrics"] = model_results[
                "validation_metrics"
            ][model_results["best_epoch"] - 1]

            # Store validation predictions and labels for confusion matrix plotting
            all_validation_preds[test_debate].extend(validation_preds)
            all_validation_labels[test_debate].extend(validation_labels)

        # Compute span-level metrics for the test split
        span_metrics = compute_metrics_span_level(preds, labels)
        model_results["test_metrics"]["span"] = span_metrics

        # Print test metrics for the current debate
        console.print(
            f"Test | "
            f"F1={test_metrics['macro']['f1']:.2%} "
            f"P={test_metrics['macro']['precision']:.1%} "
            f"R={test_metrics['macro']['recall']:.1%} "
            f"B-F1={test_metrics['B']['f1']:.1%} "
            f"B-P={test_metrics['B']['precision']:.1%} "
            f"B-R={test_metrics['B']['recall']:.1%} "
            f"I-F1={test_metrics['I']['f1']:.1%} "
            f"I-P={test_metrics['I']['precision']:.1%} "
            f"I-R={test_metrics['I']['recall']:.1%} "
            f"O-F1={test_metrics['O']['f1']:.1%} "
            f"S-F1={span_metrics['f1']:.1%} "
            f"J={test_metrics['macro']['jaccard']:.1%}"
        )

        # Report to Optuna trial if provided
        if trial is not None:
            # Report the mean macro F1 score across all completed folds to Optuna for pruning decisions
            cumulative_mean_macro_f1 = np.array(
                [
                    mr["best_validation_metrics"]["macro"]["f1"]
                    for mr in results.values()
                ]
            ).mean()
            trial.report(cumulative_mean_macro_f1, i)
            console.print(
                f"Trial report: Cumulative mean M-F1={cumulative_mean_macro_f1:.2%}"
            )

            # Check if the trial should be pruned
            if trial.should_prune():
                trial.set_user_attr(
                    "partial_trial_data",
                    {
                        "hparams": hparams,
                        "results": results,
                        "deciding_metric": float(cumulative_mean_macro_f1),
                        "pruned": True,
                        "completed_folds": i + 1,
                        "pruned_on_debate": test_debate,
                    },
                )
                console.print(f"Trial pruned at fold {i + 1} for debate {test_debate}")
                progress.remove_task(progress_folds)
                raise TrialPruned()

        del (
            model,
            train_loader,
            test_loader,
            train_dataset,
            test_dataset,
            preds,
            labels,
            best_model_state,
        )  # Free up memory
        progress.advance(progress_folds)
        gc.collect()
        torch.cuda.empty_cache()
    progress.remove_task(progress_folds)
    if not trial:
        progress.stop()

    if model_output_dir is not None:
        with (model_output_dir / f"test_preds_labels.json").open("w") as f:
            json.dump(
                {"preds": all_test_preds, "labels": all_test_labels},
                f,
                ensure_ascii=False,
                indent=4,
            )
        if val:
            with (model_output_dir / f"validation_preds_labels.json").open("w") as f:
                json.dump(
                    {"preds": all_validation_preds, "labels": all_validation_labels},
                    f,
                    ensure_ascii=False,
                    indent=4,
                )

    results["overall"] = {"test": {}}
    if val:
        results["overall"]["validation"] = {}

    # Compile best epochs and compute median best epoch across debates
    results["overall"]["best_epochs"] = [
        results[debate]["best_epoch"] for debate in all_debates
    ]
    results["overall"]["best_epoch_median"] = int(
        np.median(results["overall"]["best_epochs"])
    )

    # Compute average metrics across debates for each label and metric
    for label in ["macro", "span", "B", "I", "O"]:
        # Initialize overall metrics dictionaries for each label
        results["overall"]["test"][label] = {}
        if val:
            results["overall"]["validation"][label] = {}

        for metric in ["f1", "precision", "recall"]:
            # Compute average test metric across debates for the current label and metric
            results["overall"]["test"][label][metric] = np.mean(
                [
                    results[debate]["test_metrics"][label][metric]
                    for debate in all_debates
                ]
            )

            # Compute average validation metric across debates for the
            # current label and metric if validation was performed
            if val:
                results["overall"]["validation"][label][metric] = np.mean(
                    [
                        results[debate]["best_validation_metrics"][label][metric]
                        for debate in all_debates
                    ]
                )

    # Jaccard score is only computed for the macro label, so we compute it separately
    if val:
        results["overall"]["validation"]["macro"]["jaccard"] = np.mean(
            [
                results[debate]["best_validation_metrics"]["macro"]["jaccard"]
                for debate in all_debates
            ]
        )

    results["overall"]["test"]["macro"]["jaccard"] = np.mean(
        [results[debate]["test_metrics"]["macro"]["jaccard"] for debate in all_debates]
    )

    return (
        results,
        all_test_preds,
        all_test_labels,
        all_validation_preds,
        all_validation_labels,
    )


# ---------------------------------
# Results Processing and Plotting
# ---------------------------------


def print_overall_results(results: dict) -> None:
    # Print overall token-level test metrics
    console.print(
        f"\nOverall test metrics: "
        f"F1={results['overall']['test']['macro']['f1']:.2%} "
        f"P={results['overall']['test']['macro']['precision']:.1%} "
        f"R={results['overall']['test']['macro']['recall']:.1%} "
        f"B-F1={results['overall']['test']['B']['f1']:.1%} "
        f"B-P={results['overall']['test']['B']['precision']:.1%} "
        f"B-R={results['overall']['test']['B']['recall']:.1%} "
        f"I-F1={results['overall']['test']['I']['f1']:.1%} "
        f"I-P={results['overall']['test']['I']['precision']:.1%} "
        f"I-R={results['overall']['test']['I']['recall']:.1%} "
        f"O-F1={results['overall']['test']['O']['f1']:.1%} "
        f"J={results['overall']['test']['macro']['jaccard']:.1%}"
    )

    # Print overall span-level test metrics
    console.print(
        f"Overall span-level test metrics: "
        f"F1={results['overall']['test']['span']['f1']:.2%} "
        f"P={results['overall']['test']['span']['precision']:.1%} "
        f"R={results['overall']['test']['span']['recall']:.1%} "
    )

    # Print overall token-level validation metrics
    if results["overall"].get("validation") is not None:
        console.print(
            f"Overall validation metrics: "
            f"F1={results['overall']['validation']['macro']['f1']:.2%} "
            f"P={results['overall']['validation']['macro']['precision']:.1%} "
            f"R={results['overall']['validation']['macro']['recall']:.1%} "
            f"B-F1={results['overall']['validation']['B']['f1']:.1%} "
            f"B-P={results['overall']['validation']['B']['precision']:.1%} "
            f"B-R={results['overall']['validation']['B']['recall']:.1%} "
            f"I-F1={results['overall']['validation']['I']['f1']:.1%} "
            f"I-P={results['overall']['validation']['I']['precision']:.1%} "
            f"I-R={results['overall']['validation']['I']['recall']:.1%} "
            f"O-F1={results['overall']['validation']['O']['f1']:.1%} "
            f"S-F1={results['overall']['validation']['span']['f1']:.1%} "
            f"J={results['overall']['validation']['macro']['jaccard']:.1%} "
        )


def process_results(
    results: dict,
    output_dir: Path,
    test_preds: list,
    test_labels: list,
    val_preds: list,
    val_labels: list,
) -> None:
    console.rule("Final Results and Plots")
    # Save results to JSON
    with (output_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    console.print(f"Results saved to {output_dir / 'results.json'}")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Print per-left-out-debate results
    console.print("\nPer-left-out-debate test results:")
    for debate, metrics in results.items():
        if debate == "overall":
            continue
        console.print(
            f"Debate '{debate}': M-F1 = {metrics['test_metrics']['macro']['f1']:.1%}"
        )

    print_overall_results(results)

    if val_preds:
        # Plot training and validation loss curves
        plot_train_val_loss_curves(
            results, figures_dir / "training_validation_loss_curves.png"
        )
        # Plot validation metrics
        plot_validation_metric_curves(results, figures_dir)
        # Plot validation confusion matrix
        plot_confusion_matrix(
            val_preds,
            val_labels,
            figures_dir / "validation_confusion_matrix.png",
            normalize=True,
        )
    else:
        # Plot training loss curve
        plot_train_loss_curve(results, figures_dir / "training_loss_curve.png")

    # Plot test confusion matrix
    plot_confusion_matrix(
        test_preds,
        test_labels,
        figures_dir / "test_confusion_matrix.png",
        normalize=True,
    )

    console.print(f"\nPlots saved to '{figures_dir}'")


# -------------------------------
# Argument Parsing and Main Function
# -------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("input_file", type=str, help="Path to input CSV file")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="Name of the dataset (for saving models)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_DEFAULT,
        help="Name of the transformer model to use.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate for the model."
    )
    parser.add_argument(
        "--num-epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for training."
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--lr-crf-mult",
        type=float,
        default=LEARNING_RATE_CRF_MULTIPLIER,
        help="Learning rate multiplier for the CRF layer (if used).",
    )
    parser.add_argument(
        "--lr-fc-mult",
        type=float,
        default=LEARNING_RATE_FC_MULTIPLIER,
        help="Learning rate multiplier for the fully connected layer (if used).",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate scheduler.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="Number of initial transformer layers to freeze during training.",
    )
    parser.add_argument(
        "--crf-priors",
        action="store_true",
        help="Whether to use CRF priors for BIO sequence modeling.",
    )
    parser.add_argument(
        "--emission-bias",
        action="store_true",
        help="Whether to use emission bias for CRF layer.",
    )
    parser.add_argument(
        "--mixed-precision-dtype",
        type=str,
        choices=["fp16", "bf16", "none"],
        default="bf16",
        help="Data type for mixed precision training.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Whether to enable gradient checkpointing for memory efficiency.",
    )
    parser.add_argument(
        "--comment",
        type=str,
        default="",
        help="Optional comment to include in the saved hyperparameters for context.",
    )
    parser.add_argument(
        "--no-crf",
        action="store_false",
        help="Whether to disable the CRF layer on top of the transformer model.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Whether to save the trained models to disk.",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="Whether to perform validation during training (enables early stopping).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help="Number of epochs with no improvement after which training will be stopped.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main():
    torch.cuda.empty_cache()
    torch.set_default_dtype(torch.float32)
    args = parse_args()
    set_random_seed(args.seed)
    if args.dataset_name == "":
        args.dataset_name = Path(args.input_file).name.split(".")[0].split("_")[0]

    df = pd.read_csv(args.input_file)
    console.print(f"Loaded data with {len(df)} rows from {args.input_file}.")

    model_output_dir = get_model_output_dir(
        args.dataset_name, args.model_name, args.comment
    )
    model_output_dir.mkdir(exist_ok=True, parents=True)

    hparams = {
        "dropout": args.dropout,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "lr": args.learning_rate,
        "lr_fc_mult": args.lr_fc_mult,
        "lr_crf_mult": args.lr_crf_mult,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "use_crf": args.no_crf,
        "debate_alpha": args.debate_alpha,
        "bio_alpha": args.bio_alpha,
        "bio_eps": args.bio_eps,
        "crf_priors": args.crf_priors,
        "emission_bias": args.emission_bias,
        "mixed_precision_dtype": args.mixed_precision_dtype,
        "gradient_checkpointing": args.gradient_checkpointing,
        "freeze": args.freeze,
        "seed": args.seed,
        "patience": args.patience,
        "comment": args.comment,
    }

    results, test_preds, test_labels, val_preds, val_labels = train_lodo(
        df, args.model_name, hparams, args.val, model_output_dir, args.save
    )

    process_results(
        results, model_output_dir, test_preds, test_labels, val_preds, val_labels
    )


if __name__ == "__main__":
    main()
