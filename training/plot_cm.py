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
    all_preds, all_labels, output_path: Path, normalize: bool = True
):
    all_preds = deepcopy(all_preds)
    all_labels = deepcopy(all_labels)
    console.print("Plotting confusion matrix...")
    y_true = []
    y_pred = []

    assert len(all_labels) == len(
        all_preds
    ), "Number of sequences in predictions and labels must be the same."
    for i in range(len(all_preds)):
        all_labels[i] = all_labels[i][
            : len(all_preds[i])
        ]  # Truncate labels to match the length of predictions (remove padding)
        all_labels[i] = [
            id2label[label_id] for label_id in all_labels[i]
        ]  # Convert label IDs to label names
        all_preds[i] = [
            id2label[label_id] for label_id in all_preds[i]
        ]  # Convert prediction IDs to label names
        y_true.extend(all_labels[i])
        y_pred.extend(all_preds[i])

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
    plt.savefig(output_path, dpi=300)
    plt.close()
    gc.collect()


def compute_metrics_span_level(all_preds: list, all_labels: list) -> dict:
    all_preds = deepcopy(all_preds)
    all_labels = deepcopy(all_labels)
    id2label_seqeval = deepcopy(id2label)

    if len(label_list) == 3:
        # Handle the case where there is only one unnamed entity (B, I)
        id2label_seqeval[label2id["B"]] = "B-CLAIM"
        id2label_seqeval[label2id["I"]] = "I-CLAIM"

    assert len(all_labels) == len(
        all_preds
    ), "Number of sequences in predictions and labels must be the same."
    for i in range(len(all_preds)):
        all_labels[i] = all_labels[i][
            : len(all_preds[i])
        ]  # Truncate labels to match the length of predictions (remove padding)
        all_labels[i] = [
            id2label_seqeval[label_id] for label_id in all_labels[i]
        ]  # Convert label IDs to label names
        all_preds[i] = [
            id2label_seqeval[label_id] for label_id in all_preds[i]
        ]  # Convert prediction IDs to label names

    report = classification_report(
        all_labels, all_preds, output_dict=True, zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)
    del all_labels, all_preds, id2label_seqeval
    gc.collect()
    return {
        "f1": report["macro avg"]["f1-score"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "accuracy": accuracy,
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
    test_or_val = "validation" if "validation" in args.all_preds_labels_file else "test"

    plot_confusion_matrix(
        all_preds_labels["preds"],
        all_preds_labels["labels"],
        figures_dir / f"{test_or_val}_confusion_matrix.png",
        normalize=True,
    )
    span_metrics = compute_metrics_span_level(
        all_preds_labels["preds"], all_preds_labels["labels"]
    )
    console.print(
        f"Span metrics: "
        f"F1={span_metrics['f1']:.1%}, "
        f"A={span_metrics['accuracy']:.1%}, "
        f"P={span_metrics['precision']:.1%}, "
        f"R={span_metrics['recall']:.1%}"
    )


if __name__ == "__main__":
    main()
