import gc
import sys

import pandas as pd
import torch

from rich.console import Console
from rich.progress import (
    MofNCompleteColumn,
    TimeElapsedColumn,
    Progress,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    Column,
)

MODEL_DEFAULT = "distilroberta-base"

CRF_LEARNING_RATE_MULTIPLIER = 10
DEBATE_TEMPERED_SAMPLING_ALPHA = 1.0  # 0: Debates are weighted equally / 1: Original
BIO_TEMPERED_SAMPLING_ALPHA = 0.0  # 0: Original / 1: Linear weighting of BIO
BIO_TEMPERED_SAMPLING_EPS = 1.0

EMISSION_BIAS_B = 0.75
EMISSION_BIAS_I = 0.25

# Label mapping
label_list = ["O", "B", "I"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

console = Console()


def handle_interrupt(signum, frame):
    # console.print("\nKeyboardInterrupt detected. Running garbage collection...")

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    sys.exit(0)


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


def decay_group(module):
    return [
        p
        for n, p in module.named_parameters()
        if p.requires_grad and "bias" not in n and "LayerNorm.weight" not in n
    ]


def no_decay_group(module):
    return [
        p
        for n, p in module.named_parameters()
        if p.requires_grad and ("bias" in n or "LayerNorm.weight" in n)
    ]


def get_optimizer(model, hparams):
    optimizer_params = [
        # Transformer
        {
            "params": decay_group(model.transformer),
            "lr": hparams["lr"],
            "weight_decay": hparams["weight_decay"],
        },
        {
            "params": no_decay_group(model.transformer),
            "lr": hparams["lr"],
            "weight_decay": 0.0,
        },
        # FC layer
        {
            "params": decay_group(model.fc),
            "lr": 1e-3,
            "weight_decay": hparams["weight_decay"],
        },
        {
            "params": no_decay_group(model.fc),
            "lr": 1e-3,
            "weight_decay": 0.0,
        },
    ]
    if model.use_crf:
        optimizer_params.extend(
            [
                {
                    "params": decay_group(model.crf),
                    "lr": 1e-3 if not hparams["crf_priors"] else 5e-4,
                    "weight_decay": 0.0,  # IMPORTANT: no decay on CRF
                },
                {
                    "params": no_decay_group(model.crf),
                    "lr": 1e-3 if not hparams["crf_priors"] else 5e-4,
                    "weight_decay": 0.0,
                },
            ]
        )
    return torch.optim.AdamW(optimizer_params)


def create_progress_bar(console) -> Progress:
    """
    Create a rich progress bar with a spinner and time remaining.
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(table_column=Column(justify="right", width=10)),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        transient=True,
        console=console,
    )
    return progress


progress = create_progress_bar(console)
