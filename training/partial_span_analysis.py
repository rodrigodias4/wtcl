import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from utils import console, id2label


def bio_sequence_to_spans(labels: Sequence) -> list[tuple[int, int]]:
    """
    Convert a BIO sequence into spans over token positions.

    The training artifacts store BIO labels per token, but they do not include
    character offsets, so we use token-index spans to support IoU-based matching.
    """
    spans = []
    start = None

    for idx, label in enumerate(labels):
        label_type = id2label[label]

        if label_type == "B":
            if start is not None:
                spans.append((start, idx))
            start = idx
        elif label_type == "I":
            if start is None:
                start = idx
        else:
            if start is not None:
                spans.append((start, idx))
                start = None

    if start is not None:
        spans.append((start, len(labels)))

    return spans


def span_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Compute IoU between two half-open spans [start, end)."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter == 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union


def compute_partial_span_metrics(
    gold_sequences: Iterable[Sequence],
    pred_sequences: Iterable[Sequence],
    thresholds: Sequence[float],
) -> list[dict]:
    metrics = []

    gold_sequences = list(gold_sequences)
    pred_sequences = list(pred_sequences)

    if len(gold_sequences) != len(pred_sequences):
        raise ValueError(
            f"Number of gold sequences ({len(gold_sequences)}) does not match predictions ({len(pred_sequences)})."
        )

    for threshold in thresholds:
        tp = 0
        fp = 0
        fn = 0

        for gold_seq, pred_seq in zip(gold_sequences, pred_sequences):
            gold_spans = bio_sequence_to_spans(gold_seq)
            pred_spans = bio_sequence_to_spans(pred_seq)

            if not gold_spans:
                fp += len(pred_spans)
                continue

            if not pred_spans:
                fn += len(gold_spans)
                continue

            cost = np.zeros((len(gold_spans), len(pred_spans)))
            for i, gold_span in enumerate(gold_spans):
                for j, pred_span in enumerate(pred_spans):
                    cost[i, j] = -span_iou(gold_span, pred_span)

            rows, cols = linear_sum_assignment(cost)
            matched_gold = set()
            matched_pred = set()

            for row, col in zip(rows, cols):
                iou = -cost[row, col]
                if iou >= threshold:
                    tp += 1
                    matched_gold.add(row)
                    matched_pred.add(col)

            fn += len(gold_spans) - len(matched_gold)
            fp += len(pred_spans) - len(matched_pred)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )

        metrics.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    return metrics


def plot_partial_span_metrics(metrics: Sequence[dict], output_path: Path) -> None:
    thresholds = [m["threshold"] for m in metrics]
    precision = [m["precision"] for m in metrics]
    recall = [m["recall"] for m in metrics]
    f1 = [m["f1"] for m in metrics]

    plt.figure(figsize=(7, 4.5))
    plt.plot(thresholds, precision, marker="o", label="Precision")
    plt.plot(thresholds, recall, marker="o", label="Recall")
    plt.plot(thresholds, f1, marker="o", label="F1")
    plt.xticks(thresholds)
    plt.xlabel("IoU threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute partial span IoU metrics from model training predictions and labels."
    )
    parser.add_argument(
        "preds_labels_file",
        type=str,
        help="Path to the JSON file produced by training that contains 'preds' and 'labels'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the plot. Defaults to a 'figures' directory next to the input file.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=None,
        help="Optional IoU thresholds to evaluate. Defaults to 0.1 to 1.0 in 0.1 steps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.preds_labels_file)
    with input_path.open("r") as handle:
        payload = json.load(handle)

    preds = payload.get("preds", {})
    labels = payload.get("labels", {})
    if not preds or not labels:
        raise ValueError(
            "The input JSON file must contain non-empty 'preds' and 'labels' objects."
        )

    if set(preds.keys()) != set(labels.keys()):
        raise ValueError(
            "The 'preds' and 'labels' objects must contain the same debate keys."
        )

    gold_sequences = []
    pred_sequences = []
    for debate in sorted(preds):
        debate_preds = preds[debate]
        debate_labels = labels[debate]
        if len(debate_preds) != len(debate_labels):
            raise ValueError(
                f"Debate '{debate}' has {len(debate_preds)} predictions and {len(debate_labels)} labels."
            )
        for pred_seq, gold_seq in zip(debate_preds, debate_labels):
            pred_sequences.append(pred_seq)
            gold_sequences.append(gold_seq)

    thresholds = args.thresholds
    if thresholds is None:
        thresholds = [round(x, 1) for x in np.arange(0.1, 1.0 + 1e-9, 0.1)]

    metrics = compute_partial_span_metrics(gold_sequences, pred_sequences, thresholds)

    output_dir = (
        Path(args.output_dir) if args.output_dir else input_path.parent / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "partial_span_iou_curve.png"
    plot_partial_span_metrics(metrics, plot_path)

    console.print("Partial span metrics by IoU threshold:")
    for entry in metrics:
        console.print(
            f"IoU ≥ {entry['threshold']:.1f}: "
            f"F1={entry['f1'] * 100:.2f}  "
            f"Precision={entry['precision'] * 100:.2f}  "
            f"Recall={entry['recall'] * 100:.2f}"
        )
    console.print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
