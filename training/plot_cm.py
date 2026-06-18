from copy import deepcopy
import gc
import json
from pathlib import Path

import numpy as np

from utils import label_list, id2label, label2id, console
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from seqeval.metrics import accuracy_score, classification_report


def plot_confusion_matrix(
    all_preds_labels: dict, output_dir: Path, normalize: bool = True
):
    console.print("Plotting confusion matrix...")
    y_true_seqs = deepcopy(all_preds_labels["labels"])
    y_pred_seqs = deepcopy(all_preds_labels["preds"])
    y_true = []
    y_pred = []

    assert len(y_true_seqs) == len(
        y_pred_seqs
    ), "Number of sequences in predictions and labels must be the same."
    for i in range(len(y_pred_seqs)):
        y_true_seqs[i] = y_true_seqs[i][
            : len(y_pred_seqs[i])
        ]  # Truncate labels to match the length of predictions (remove padding)
        y_true_seqs[i] = [
            id2label[label_id] for label_id in y_true_seqs[i]
        ]  # Convert label IDs to label names
        y_pred_seqs[i] = [
            id2label[label_id] for label_id in y_pred_seqs[i]
        ]  # Convert prediction IDs to label names
        y_true.extend(y_true_seqs[i])
        y_pred.extend(y_pred_seqs[i])

    gc.collect()

    cm = confusion_matrix(
        y_true, y_pred, labels=label_list, normalize="true" if normalize else None
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="BuPu",
        xticklabels=label_list,
        yticklabels=label_list,
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("BIO Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()
    gc.collect()


def compute_metrics_span_level(all_preds_labels: dict) -> dict:
    y_true_seqs = deepcopy(all_preds_labels["labels"])
    y_pred_seqs = deepcopy(all_preds_labels["preds"])

    id2label_seqeval = deepcopy(id2label)

    if len(label_list) == 3:
        # Handle the case where there is only one unnamed entity (B, I)
        id2label_seqeval[label2id["B"]] = "B-CLAIM"
        id2label_seqeval[label2id["I"]] = "I-CLAIM"

    assert len(y_true_seqs) == len(
        y_pred_seqs
    ), "Number of sequences in predictions and labels must be the same."
    for i in range(len(y_pred_seqs)):
        y_true_seqs[i] = y_true_seqs[i][
            : len(y_pred_seqs[i])
        ]  # Truncate labels to match the length of predictions (remove padding)
        y_true_seqs[i] = [
            id2label_seqeval[label_id] for label_id in y_true_seqs[i]
        ]  # Convert label IDs to label names
        y_pred_seqs[i] = [
            id2label_seqeval[label_id] for label_id in y_pred_seqs[i]
        ]  # Convert prediction IDs to label names

    report = classification_report(
        y_true_seqs, y_pred_seqs, output_dict=True, zero_division=0
    )
    gc.collect()
    return {
        "f1": report["macro avg"]["f1-score"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "accuracy": accuracy_score(y_true_seqs, y_pred_seqs),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Plot confusion matrix.")
    parser.add_argument(
        "all_preds_labels_file",
        type=str,
        help="Path to the JSON file containing all predictions and labels",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.all_preds_labels_file, "r") as f:
        all_preds_labels = json.load(f)

    figures_dir = Path(args.all_preds_labels_file).parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    plot_confusion_matrix(all_preds_labels, figures_dir, normalize=True)
    span_metrics = compute_metrics_span_level(all_preds_labels)
    console.print(
        f"Span metrics: "
        f"F1={span_metrics['f1']:.3f}"
        f"A={span_metrics['accuracy']:.3f}, "
        f"P={span_metrics['precision']:.3f}, "
        f"R={span_metrics['recall']:.3f}, "
    )


if __name__ == "__main__":
    main()
