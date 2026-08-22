import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from partial_span_analysis import compute_partial_span_metrics
from plot_cm import compute_metrics_span_level
from utils import console

TAG_COLUMNS = {
    "form": "reason_form",
    "frame": "reason_frame",
    "domain": "reason_domain",
}
PARTIAL_THRESHOLDS = (0.25, 0.5, 0.75)


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


def _tagged_sequences(
    df: pd.DataFrame,
    labels: dict[str, list[Sequence]],
    predictions: dict[str, list[Sequence]],
) -> dict[str, dict[str, tuple[list[Sequence], list[Sequence]]]]:
    tagged = {dimension: {} for dimension in TAG_COLUMNS}
    debate_column = "debate_id" if "debate_id" in df.columns else "debate"

    for debate in sorted(labels):
        debate_df = df[df[debate_column] == debate].sort_values("id")
        debate_labels = labels[debate]
        debate_predictions = predictions[debate]
        if len(debate_df) != len(debate_labels) or len(debate_labels) != len(
            debate_predictions
        ):
            raise ValueError(
                f"Debate {debate} has mismatched dataset, label, and prediction lengths."
            )

        for sequence_index, (_, row) in enumerate(debate_df.iterrows()):
            spans = _read_spans(row["spans"])
            for dimension, span_key in TAG_COLUMNS.items():
                tags = {tag for span in spans for tag in span.get(span_key, [])}
                for tag in tags:
                    gold, pred = tagged[dimension].setdefault(tag, ([], []))
                    gold.append(debate_labels[sequence_index])
                    pred.append(debate_predictions[sequence_index])

    return tagged


def compute_metrics_by_tag(
    df: pd.DataFrame,
    labels: dict[str, list[Sequence]],
    predictions: dict[str, list[Sequence]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    tagged = _tagged_sequences(df, labels, predictions)
    results = {dimension: {} for dimension in TAG_COLUMNS}

    for dimension, tag_sequences in tagged.items():
        for tag, (gold, pred) in sorted(tag_sequences.items()):
            exact = compute_metrics_span_level(pred, gold)["f1"]
            partial_metrics = compute_partial_span_metrics(
                gold, pred, PARTIAL_THRESHOLDS
            )
            results[dimension][tag] = {
                "exact_match": exact,
                "partial_span_f1": {
                    threshold: metric["f1"]
                    for threshold, metric in zip(PARTIAL_THRESHOLDS, partial_metrics)
                },
                "count": len(gold),
            }

    return results


def plot_metrics_by_tag(
    metrics: dict[str, dict[str, dict[str, float | int]]], output_dir: Path
) -> list[Path]:
    output_paths = []
    for dimension in TAG_COLUMNS:
        figure, axis = plt.subplots(figsize=(10, 6))
        tags = sorted(
            metrics[dimension],
            key=lambda tag: metrics[dimension][tag]["partial_span_f1"][0.5],
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

        axis.scatter(
            x_values, exact, label="Exact-match F1", marker="o", color="black", zorder=2
        )
        for threshold, color in zip(
            PARTIAL_THRESHOLDS, ("tab:blue", "tab:orange", "tab:green")
        ):
            partial = [
                metrics[dimension][tag]["partial_span_f1"][threshold] for tag in tags
            ]
            axis.scatter(
                x_values,
                partial,
                label=f"Partial span F1 (IoU >= {threshold})",
                marker="o",
                color=color,
                zorder=3,
            )
        axis.set_xticks(x_values, tags, rotation=45, ha="right")
        axis.set_ylim(0, 1)
        axis.set_ylabel("F1")
        axis.grid(alpha=0.3, zorder=0)
        axis.set_zorder(2)
        axis.patch.set_visible(False)

        axis.legend()
        figure.tight_layout()
        output_path = output_dir / f"performance_by_tag_{dimension}.png"
        figure.savefig(output_path, dpi=300)
        plt.close(figure)
        output_paths.append(output_path)

    return output_paths


def main() -> None:
    args = parse_args()
    with args.labels_predictions_path.open() as handle:
        payload = json.load(handle)

    labels = payload["labels"]
    predictions = payload.get("predictions", payload.get("preds"))
    if predictions is None:
        raise ValueError("The JSON file must contain 'predictions' or 'preds'.")

    df = pd.read_csv(args.dataset_path).sort_values("id")
    metrics = compute_metrics_by_tag(df, labels, predictions)

    output_dir = args.output_dir or args.labels_predictions_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = plot_metrics_by_tag(metrics, output_dir)

    for dimension, dimension_metrics in metrics.items():
        for tag, values in dimension_metrics.items():
            console.print(
                f"{dimension}/{tag}: n={values['count']} "
                f"partial@0.25={values['partial_span_f1'][0.25]:.3f} "
                f"partial@0.5={values['partial_span_f1'][0.5]:.3f}"
                f"partial@0.75={values['partial_span_f1'][0.75]:.3f} "
                f"exact={values['exact_match']:.3f} "
            )
    for output_path in output_paths:
        console.print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
