import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import linear_sum_assignment
import numpy as np

from partial_span_analysis import (
    bio_sequence_to_spans,
    span_iou,
)
from utils import console

TAG_COLUMNS = {
    "form": "reason_form",
    "frame": "reason_frame",
    "domain": "reason_domain",
}
PARTIAL_THRESHOLDS = (0.25, 0.5, 0.75)
FIGSIZES = {
    "form": (5, 4),
    "frame": (5, 4),
    "domain": (9, 5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot exact and partial span performance by annotation tag."
    )
    parser.add_argument(
        "labels_predictions_path",
        type=Path,
        help="Path to the JSON file containing model predictions and labels.",
    )
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the plot. Defaults to a figures directory next to the predictions.",
    )
    return parser.parse_args()


def _read_spans(value: str) -> list[dict]:
    spans = json.loads(value)
    if not isinstance(spans, list):
        raise ValueError("The spans column must contain a JSON list.")
    return spans


def _match_gold_claims(
    gold_spans: list[tuple[int, int]],
    predicted_spans: list[tuple[int, int]],
) -> dict[int, dict[float, bool]]:
    """
    Match gold claims to predictions using the same one-to-one
    Hungarian matching strategy as partial_span_analysis.py.

    Returns, for each gold claim, whether it was detected at each
    IoU threshold.
    """
    matches = {
        i: {threshold: False for threshold in (1.0, *PARTIAL_THRESHOLDS)}
        for i in range(len(gold_spans))
    }

    if not gold_spans or not predicted_spans:
        return matches

    cost = np.zeros((len(gold_spans), len(predicted_spans)))

    for i, gold_span in enumerate(gold_spans):
        for j, pred_span in enumerate(predicted_spans):
            cost[i, j] = -span_iou(gold_span, pred_span)

    rows, cols = linear_sum_assignment(cost)

    for row, col in zip(rows, cols):
        iou = -cost[row, col]

        for threshold in (1.0, *PARTIAL_THRESHOLDS):
            if iou >= threshold:
                matches[row][threshold] = True

    return matches


# REPLACE compute_metrics_by_tag() WITH THIS


def compute_metrics_by_tag(
    df: pd.DataFrame,
    labels: dict[str, list[Sequence]],
    predictions: dict[str, list[Sequence]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """
    Compute claim-level recall conditioned on annotation tags.

    Each gold claim contributes to every tag it carries. A claim is
    considered detected when it is matched to a predicted span at the
    relevant IoU threshold.

    False-positive predictions are not assigned to tags because they
    have no gold annotation from which to obtain a tag.
    """
    debate_column = "debate_id" if "debate_id" in df.columns else "debate"

    counts = {dimension: {} for dimension in TAG_COLUMNS}

    for debate in sorted(labels):
        debate_df = df[df[debate_column] == debate].sort_values("id")
        debate_labels = labels[debate]
        debate_predictions = predictions[debate]

        if len(debate_df) != len(debate_labels):
            raise ValueError(
                f"Debate {debate} has mismatched dataset and label lengths."
            )

        if len(debate_labels) != len(debate_predictions):
            raise ValueError(
                f"Debate {debate} has mismatched label and prediction lengths."
            )

        for sequence_index, (_, row) in enumerate(debate_df.iterrows()):
            annotation_spans = _read_spans(row["spans"])

            gold_spans = bio_sequence_to_spans(debate_labels[sequence_index])
            predicted_spans = bio_sequence_to_spans(debate_predictions[sequence_index])

            if len(annotation_spans) != len(gold_spans):
                raise ValueError(
                    f"Debate {debate}, row id={row['id']}: "
                    f"{len(annotation_spans)} annotation spans but "
                    f"{len(gold_spans)} BIO gold spans."
                )

            matches = _match_gold_claims(
                gold_spans,
                predicted_spans,
            )

            for claim_index, annotation_span in enumerate(annotation_spans):
                for dimension, span_key in TAG_COLUMNS.items():
                    tags = set(annotation_span.get(span_key, []))

                    for tag in tags:
                        if tag not in counts[dimension]:
                            counts[dimension][tag] = {
                                "count": 0,
                                "exact_tp": 0,
                                "partial_tp": {
                                    threshold: 0 for threshold in PARTIAL_THRESHOLDS
                                },
                            }

                        counts[dimension][tag]["count"] += 1

                        if matches[claim_index][1.0]:
                            counts[dimension][tag]["exact_tp"] += 1

                        for threshold in PARTIAL_THRESHOLDS:
                            if matches[claim_index][threshold]:
                                counts[dimension][tag]["partial_tp"][threshold] += 1

    results = {dimension: {} for dimension in TAG_COLUMNS}

    for dimension, tag_counts in counts.items():
        for tag, values in sorted(tag_counts.items()):
            if values["count"] < 5:
                console.print(
                    f"Skipping {dimension}/{tag} because it has fewer than 5 claims."
                )
                continue

            count = values["count"]

            results[dimension][tag] = {
                "exact_match": values["exact_tp"] / count,
                "partial_span": {
                    threshold: values["partial_tp"][threshold] / count
                    for threshold in PARTIAL_THRESHOLDS
                },
                "count": count,
            }

    return results


# ADD THIS FUNCTION BEFORE main()


def compute_overall_claim_recall(
    df: pd.DataFrame,
    labels: dict[str, list[Sequence]],
    predictions: dict[str, list[Sequence]],
) -> dict:
    """
    Compute overall claim-level recall using the same one-to-one
    matching procedure as the per-tag analysis.
    """
    debate_column = "debate_id" if "debate_id" in df.columns else "debate"

    total_claims = 0
    exact_tp = 0
    partial_tp = {threshold: 0 for threshold in PARTIAL_THRESHOLDS}

    for debate in sorted(labels):
        debate_df = df[df[debate_column] == debate].sort_values("id")
        debate_labels = labels[debate]
        debate_predictions = predictions[debate]

        for sequence_index, (_, row) in enumerate(debate_df.iterrows()):
            annotation_spans = _read_spans(row["spans"])

            gold_spans = bio_sequence_to_spans(debate_labels[sequence_index])
            predicted_spans = bio_sequence_to_spans(debate_predictions[sequence_index])

            if len(annotation_spans) != len(gold_spans):
                raise ValueError(
                    f"Debate {debate}, row id={row['id']}: "
                    f"{len(annotation_spans)} annotation spans but "
                    f"{len(gold_spans)} BIO gold spans."
                )

            matches = _match_gold_claims(
                gold_spans,
                predicted_spans,
            )

            total_claims += len(gold_spans)

            for claim_index in range(len(gold_spans)):
                if matches[claim_index][1.0]:
                    exact_tp += 1

                for threshold in PARTIAL_THRESHOLDS:
                    if matches[claim_index][threshold]:
                        partial_tp[threshold] += 1

    return {
        "exact_match": exact_tp / total_claims if total_claims else 0.0,
        "partial_span": {
            threshold: (partial_tp[threshold] / total_claims if total_claims else 0.0)
            for threshold in PARTIAL_THRESHOLDS
        },
        "count": total_claims,
    }


def plot_metrics_by_tag(
    metrics: dict[str, dict[str, dict[str, float | int]]],
    output_dir: Path,
) -> list[Path]:
    output_paths = []
    for dimension in TAG_COLUMNS:
        figure, axis = plt.subplots(figsize=FIGSIZES[dimension])
        tags = sorted(
            metrics[dimension],
            key=lambda tag: metrics[dimension][tag]["count"],
            reverse=True,
        )
        x_values = list(range(len(tags)))
        exact = [metrics[dimension][tag]["exact_match"] for tag in tags]
        support = [metrics[dimension][tag]["count"] for tag in tags]

        support_axis = axis.twinx()
        support_axis.bar(
            x_values,
            support,
            width=0.8,
            color="lightgray",
            alpha=0.5,
            label="Support",
            zorder=1,
        )
        support_axis.set_ylabel("Support", color="dimgray")
        support_axis.tick_params(axis="y", colors="dimgray")
        support_axis.set_ylim(0, max(support, default=0) * 1.2 or 1)
        support_axis.set_zorder(1)
        support_axis.patch.set_visible(False)

        for i, (threshold, color) in enumerate(
            zip(PARTIAL_THRESHOLDS, ("tab:blue", "tab:orange", "tab:green"))
        ):
            partial = [
                metrics[dimension][tag]["partial_span"][threshold] for tag in tags
            ]
            axis.scatter(
                x_values,
                partial,
                label=f"Partial-span (IoU $\geqslant$ {threshold})",
                marker="o",
                color=color,
                zorder=3,
            )

            mean_recall = metrics["avg"]["partial_span"][threshold]
            axis.axhline(
                mean_recall,
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.5,
                label=f"Mean partial-span (IoU $\geqslant$ {threshold})",
                zorder=0,
            )

        axis.scatter(
            x_values, exact, label="Exact-match", marker="o", color="black", zorder=2
        )
        mean_recall = metrics["avg"]["exact_match"]
        axis.axhline(
            mean_recall,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
            label=f"Mean exact-match",
            zorder=0,
        )

        axis.set_xticks(x_values, tags, rotation=45, ha="right")
        axis.set_ylim(0, 1)
        axis.set_ylabel("Recall")
        axis.set_yticks(np.arange(0, 1.1, 0.1))
        axis.grid(alpha=0.3, zorder=0)
        axis.set_zorder(2)
        axis.patch.set_visible(False)

        if dimension == "domain":
            axis.legend(bbox_to_anchor=(1.1, 1), loc="upper left", fontsize="small")
        figure.tight_layout()
        output_path = output_dir / f"performance_by_tag_{dimension}.png"
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)

    return output_paths


def main() -> None:
    args = parse_args()
    with args.labels_predictions_path.open() as handle:
        payload = json.load(handle)

    labels = payload["labels"]
    predictions = payload["preds"]
    if predictions is None:
        raise ValueError("The JSON file must contain 'predictions' or 'preds'.")

    df = pd.read_csv(args.dataset_path).sort_values("id")

    metrics = compute_metrics_by_tag(df, labels, predictions)

    metrics["avg"] = compute_overall_claim_recall(
        df,
        labels,
        predictions,
    )

    output_dir = args.output_dir or args.labels_predictions_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = plot_metrics_by_tag(metrics, output_dir)
    console.print(metrics)
    for dimension, dimension_metrics in metrics.items():
        for tag, values in dimension_metrics.items():
            if dimension == "avg":
                continue
            console.print(
                f"{dimension}/{tag}: n={values['count']} "
                f"partial@0.25={values['partial_span'][0.25]:.3f} "
                f"partial@0.5={values['partial_span'][0.5]:.3f} "
                f"partial@0.75={values['partial_span'][0.75]:.3f} "
                f"exact={values['exact_match']:.3f} "
            )

    console.print(
        f"Overall: n={metrics['avg']['count']} "
        f"partial@0.25={metrics['avg']['partial_span'][0.25]:.3f} "
        f"partial@0.5={metrics['avg']['partial_span'][0.5]:.3f} "
        f"partial@0.75={metrics['avg']['partial_span'][0.75]:.3f} "
        f"exact={metrics['avg']['exact_match']:.3f} "
    )

    for output_path in output_paths:
        console.print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
