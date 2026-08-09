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

LEARNING_RATE_CRF_MULTIPLIER = 10.0
LEARNING_RATE_FC_MULTIPLIER = 10.0

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
        speed_estimate_period=60 * 30,
    )
    return progress


progress = create_progress_bar(console)
