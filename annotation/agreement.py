from argparse import ArgumentParser
from ast import literal_eval
import ast
from itertools import product
from pathlib import Path
import sys
from numpy import zeros
from pandas import read_csv
from sklearn.metrics import classification_report, cohen_kappa_score
from rich.console import Console

sys.path.append((Path(__file__).resolve().parent.parent / "training").as_posix())
from train import encode, get_tokenizer
from utils import label_list
from plot_cm import compute_metrics_span_level
from typing import Tuple
from scipy.optimize import linear_sum_assignment

console = Console()


def span_iou(a: dict, b: dict) -> float:
    """
    IoU between two half-open spans [start, end).
    """
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))

    if inter == 0:
        return 0.0

    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter

    return inter / union


def partial_span_f1(df_gold, df_pred, threshold=0.5):
    """
    Compute Partial Span Precision/Recall/F1 using IoU matching.
    """

    tp = 0
    fp = 0
    fn = 0

    for (_, row_gold), (_, row_pred) in zip(df_gold.iterrows(), df_pred.iterrows()):

        gold = ast.literal_eval(row_gold["spans"])
        pred = ast.literal_eval(row_pred["spans"])

        if len(gold) == 0:
            fp += len(pred)
            continue

        if len(pred) == 0:
            fn += len(gold)
            continue

        cost = zeros((len(gold), len(pred)))

        for i, g in enumerate(gold):
            for j, p in enumerate(pred):
                cost[i, j] = -span_iou(g, p)

        rows, cols = linear_sum_assignment(cost)

        matched_gold = set()
        matched_pred = set()

        for r, c in zip(rows, cols):
            iou = -cost[r, c]

            if iou >= threshold:
                tp += 1
                matched_gold.add(r)
                matched_pred.add(c)

        fn += len(gold) - len(matched_gold)
        fp += len(pred) - len(matched_pred)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "threshold": threshold,
    }


def parse_args():
    parser = ArgumentParser(description="Compute agreement metrics for annotations.")

    parser.add_argument("file_A", type=str, help="Path to the first annotation file.")

    parser.add_argument("file_B", type=str, help="Path to the second annotation file.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load annotations from the files
    df_A = read_csv(args.file_A)
    df_B = read_csv(args.file_B)

    # Ensure both dataframes have the same length
    if len(df_A) != len(df_B):
        raise ValueError("Annotation files must have the same number of entries.")

    labels_A = []
    labels_B = []
    tokenizer = get_tokenizer("FacebookAI/roberta-base")

    for idx in range(len(df_A)):
        row_A = df_A.iloc[idx]
        row_B = df_B.iloc[idx]

        spans_A = literal_eval(row_A["spans"])
        spans_B = literal_eval(row_B["spans"])

        # Encode the spans
        enc_A = encode(row_A["text"], spans_A, tokenizer)
        enc_B = encode(row_B["text"], spans_B, tokenizer)

        labels_A.append(
            [label for label, mask in zip(enc_A["labels"], enc_A["crf_mask"]) if mask]
        )
        labels_B.append(
            [label for label, mask in zip(enc_B["labels"], enc_B["crf_mask"]) if mask]
        )

    labels_A_flat = [label for sublist in labels_A for label in sublist]
    labels_B_flat = [label for sublist in labels_B for label in sublist]

    # Cohen's Kappa
    kappa = cohen_kappa_score(labels_A_flat, labels_B_flat)
    console.print(f"Cohen's Kappa: {kappa:.2%}")

    # Token-level metrics
    token_level_metrics = classification_report(
        labels_A_flat,
        labels_B_flat,
        labels=list(range(len(label_list))),
        output_dict=True,
    )
    console.print(
        f"Macro metrics: "
        f"F1 = {token_level_metrics['macro avg']['f1-score']:.2%}, "
        f"Precision = {token_level_metrics['macro avg']['precision']:.2%}, "
        f"Recall = {token_level_metrics['macro avg']['recall']:.2%}"
    )

    for id in range(len(label_list)):
        console.print(
            f"Metrics for class {label_list[id]}: "
            f"F1 = {token_level_metrics[str(id)]['f1-score']:.2%}, "
            f"Precision = {token_level_metrics[str(id)]['precision']:.2%}, "
            f"Recall = {token_level_metrics[str(id)]['recall']:.2%}"
        )

    # Exact Span F1
    exact_span = compute_metrics_span_level(labels_A, labels_B)
    console.print(
        f"Exact Span metrics: "
        f"F1 = {exact_span['f1']:.2%}, "
        f"Precision = {exact_span['precision']:.2%}, "
        f"Recall = {exact_span['recall']:.2%}"
    )

    # Partial Span F1 for different IoU thresholds
    for t in (0.25, 0.5, 0.75):
        m = partial_span_f1(df_A, df_B, threshold=t)
        console.print(
            f"IoU ≥ {t:.2f}: F1 = {float(m['f1']):.2%} P={float(m['precision']):.2%} R={(float(m['recall'])):.2%}"
        )


if __name__ == "__main__":
    main()
