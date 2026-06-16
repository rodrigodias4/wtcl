import gc
import sys

import pandas as pd
import torch

MODEL_DEFAULT = "distilroberta-base"

CRF_LEARNING_RATE_MULTIPLIER = 10

# Label mapping
label_list = ["O", "B", "I"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

def handle_interrupt(signum, frame):
    print("\nKeyboardInterrupt detected. Running garbage collection...")
    
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
                    "lr": 1e-3,
                    "weight_decay": 0.0,  # IMPORTANT: no decay on CRF
                },
                {
                    "params": no_decay_group(model.crf),
                    "lr": 1e-3,
                    "weight_decay": 0.0,
                },
            ]
        )
    return torch.optim.AdamW(optimizer_params)
