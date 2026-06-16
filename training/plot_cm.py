import gc
import json
from pathlib import Path

import numpy as np

from utils import label_list, id2label
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def plot_confusion_matrix(all_preds_labels: dict, output_dir: Path, normalize: bool = True):
    y_true_seqs = all_preds_labels["labels"]
    y_pred_seqs = all_preds_labels["preds"]
    y_true = []
    y_pred = []

    assert len(y_true_seqs) == len(y_pred_seqs), "Number of sequences in predictions and labels must be the same."
    for i in range(len(y_pred_seqs)):
        y_true_seqs[i] = y_true_seqs[i][: len(y_pred_seqs[i])]  # Truncate labels to match the length of predictions (remove padding)
        y_true_seqs[i] = [id2label[label_id] for label_id in y_true_seqs[i]]  # Convert label IDs to label names
        y_pred_seqs[i] = [id2label[label_id] for label_id in y_pred_seqs[i]]  # Convert prediction IDs to label names
        y_true.extend(y_true_seqs[i])
        y_pred.extend(y_pred_seqs[i])
    
    gc.collect()

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=label_list,
        normalize="true" if normalize else None
    )
    
    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="BuPu",
        xticklabels=label_list,
        yticklabels=label_list
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("BIO Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot confusion matrix."
    )
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
    
    plot_confusion_matrix(all_preds_labels, Path(args.all_preds_labels_file).parent, normalize=True)


if __name__ == "__main__":
    main()