import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from partial_span_analysis import bio_sequence_to_spans, span_iou
from utils import console, id2label


def span_overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Return overlap length between two half-open spans [start, end)."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def classify_matched_error(
    gold_span: tuple[int, int], pred_span: tuple[int, int]
) -> str:
    """Assign a boundary error category for a matched gold/pred span pair."""
    start_diff = pred_span[0] - gold_span[0]
    end_diff = pred_span[1] - gold_span[1]

    if start_diff == 0 and end_diff == 0:
        return "exact_match"

    if start_diff < 0 and end_diff == 0:
        return "start_boundary_too_early"
    if start_diff > 0 and end_diff == 0:
        return "start_boundary_too_late"
    if start_diff == 0 and end_diff < 0:
        return "end_boundary_too_early"
    if start_diff == 0 and end_diff > 0:
        return "end_boundary_too_late"

    if start_diff >= 0 and end_diff <= 0:
        return "span_truncation"
    if start_diff <= 0 and end_diff >= 0:
        return "span_overextension"

    if start_diff < 0 and end_diff < 0:
        return "shifted_earlier"
    if start_diff > 0 and end_diff > 0:
        return "shifted_later"

    return "boundary_misaligned"


def flatten_sequences(payload: dict[str, Any]) -> list[dict[str, Any]]:
    preds = payload.get("preds", {})
    labels = payload.get("labels", {})
    if not preds or not labels:
        raise ValueError("Input JSON must contain non-empty 'preds' and 'labels'.")

    if set(preds.keys()) != set(labels.keys()):
        raise ValueError("'preds' and 'labels' must have the same debate keys.")

    rows: list[dict[str, Any]] = []
    for debate in sorted(preds):
        debate_preds = preds[debate]
        debate_labels = labels[debate]
        if len(debate_preds) != len(debate_labels):
            raise ValueError(
                f"Debate '{debate}' has {len(debate_preds)} predictions and {len(debate_labels)} labels."
            )

        for turn_idx, (pred_seq, gold_seq) in enumerate(
            zip(debate_preds, debate_labels)
        ):
            if len(pred_seq) != len(gold_seq):
                raise ValueError(
                    f"Debate '{debate}', turn {turn_idx} has mismatched token lengths: "
                    f"pred={len(pred_seq)} gold={len(gold_seq)}."
                )

            rows.append(
                {
                    "debate": debate,
                    "turn_idx": turn_idx,
                    "pred_seq": pred_seq,
                    "gold_seq": gold_seq,
                }
            )

    return rows


def analyze_span_errors(
    rows: Iterable[dict[str, Any]],
    iou_threshold: float,
    max_examples_per_category: int,
) -> dict[str, Any]:
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError("--iou-threshold must be between 0 and 1.")

    matched_counter: Counter[str] = Counter()
    unmatched_counter: Counter[str] = Counter()
    structure_counter: Counter[str] = Counter()
    examples: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    total_gold_spans = 0
    total_pred_spans = 0
    accepted_matches = 0

    for row in rows:
        debate = row["debate"]
        turn_idx = row["turn_idx"]
        gold_spans = bio_sequence_to_spans(row["gold_seq"])
        pred_spans = bio_sequence_to_spans(row["pred_seq"])

        total_gold_spans += len(gold_spans)
        total_pred_spans += len(pred_spans)

        if not gold_spans and not pred_spans:
            continue

        iou_matrix = np.zeros((len(gold_spans), len(pred_spans)))
        overlap_matrix = np.zeros((len(gold_spans), len(pred_spans)), dtype=int)

        for gi, g in enumerate(gold_spans):
            for pi, p in enumerate(pred_spans):
                iou_matrix[gi, pi] = span_iou(g, p)
                overlap_matrix[gi, pi] = span_overlap(g, p)

        # Structural diagnostics independent of one-to-one matching.
        for gi in range(len(gold_spans)):
            overlap_count = int(np.sum(overlap_matrix[gi, :] > 0))
            if overlap_count >= 2:
                structure_counter["split_claim_prediction"] += 1
                if len(examples["split_claim_prediction"]) < max_examples_per_category:
                    overlapping_preds = [
                        pred_spans[pi]
                        for pi in range(len(pred_spans))
                        if overlap_matrix[gi, pi] > 0
                    ]
                    examples["split_claim_prediction"].append(
                        {
                            "debate": debate,
                            "turn_idx": turn_idx,
                            "gold_span": gold_spans[gi],
                            "overlapping_pred_spans": overlapping_preds,
                        }
                    )

        for pi in range(len(pred_spans)):
            overlap_count = int(np.sum(overlap_matrix[:, pi] > 0))
            if overlap_count >= 2:
                structure_counter["merged_claim_prediction"] += 1
                if len(examples["merged_claim_prediction"]) < max_examples_per_category:
                    overlapping_gold = [
                        gold_spans[gi]
                        for gi in range(len(gold_spans))
                        if overlap_matrix[gi, pi] > 0
                    ]
                    examples["merged_claim_prediction"].append(
                        {
                            "debate": debate,
                            "turn_idx": turn_idx,
                            "pred_span": pred_spans[pi],
                            "overlapping_gold_spans": overlapping_gold,
                        }
                    )

        if not gold_spans:
            unmatched_counter["spurious_claim"] += len(pred_spans)
            for pred_span in pred_spans:
                if len(examples["spurious_claim"]) < max_examples_per_category:
                    examples["spurious_claim"].append(
                        {
                            "debate": debate,
                            "turn_idx": turn_idx,
                            "pred_span": pred_span,
                        }
                    )
            continue

        if not pred_spans:
            unmatched_counter["missed_claim"] += len(gold_spans)
            for gold_span in gold_spans:
                if len(examples["missed_claim"]) < max_examples_per_category:
                    examples["missed_claim"].append(
                        {
                            "debate": debate,
                            "turn_idx": turn_idx,
                            "gold_span": gold_span,
                        }
                    )
            continue

        rows_idx, cols_idx = linear_sum_assignment(-iou_matrix)

        matched_gold: set[int] = set()
        matched_pred: set[int] = set()

        for gi, pi in zip(rows_idx, cols_idx):
            iou = float(iou_matrix[gi, pi])
            if iou < iou_threshold:
                continue

            matched_gold.add(gi)
            matched_pred.add(pi)
            accepted_matches += 1

            category = classify_matched_error(gold_spans[gi], pred_spans[pi])
            matched_counter[category] += 1

            if len(examples[category]) < max_examples_per_category:
                examples[category].append(
                    {
                        "debate": debate,
                        "turn_idx": turn_idx,
                        "gold_span": gold_spans[gi],
                        "pred_span": pred_spans[pi],
                        "iou": iou,
                    }
                )

        for gi, gold_span in enumerate(gold_spans):
            if gi in matched_gold:
                continue

            best_iou = float(np.max(iou_matrix[gi, :])) if pred_spans else 0.0
            category = "missed_claim_with_overlap" if best_iou > 0.0 else "missed_claim"
            unmatched_counter[category] += 1

            if len(examples[category]) < max_examples_per_category:
                best_pred_idx = (
                    int(np.argmax(iou_matrix[gi, :])) if pred_spans else None
                )
                best_pred_span = (
                    pred_spans[best_pred_idx] if best_pred_idx is not None else None
                )
                examples[category].append(
                    {
                        "debate": debate,
                        "turn_idx": turn_idx,
                        "gold_span": gold_span,
                        "best_pred_span": best_pred_span,
                        "best_iou": best_iou,
                    }
                )

        for pi, pred_span in enumerate(pred_spans):
            if pi in matched_pred:
                continue

            best_iou = float(np.max(iou_matrix[:, pi])) if gold_spans else 0.0
            category = (
                "spurious_claim_with_overlap" if best_iou > 0.0 else "spurious_claim"
            )
            unmatched_counter[category] += 1

            if len(examples[category]) < max_examples_per_category:
                best_gold_idx = (
                    int(np.argmax(iou_matrix[:, pi])) if gold_spans else None
                )
                best_gold_span = (
                    gold_spans[best_gold_idx] if best_gold_idx is not None else None
                )
                examples[category].append(
                    {
                        "debate": debate,
                        "turn_idx": turn_idx,
                        "pred_span": pred_span,
                        "best_gold_span": best_gold_span,
                        "best_iou": best_iou,
                    }
                )

    total_events = sum(matched_counter.values()) + sum(unmatched_counter.values())

    summary = {
        "iou_threshold": iou_threshold,
        "num_turns": len(list(rows)) if not isinstance(rows, list) else len(rows),
        "total_gold_spans": total_gold_spans,
        "total_pred_spans": total_pred_spans,
        "accepted_matches": accepted_matches,
        "matched_categories": dict(matched_counter),
        "unmatched_categories": dict(unmatched_counter),
        "structure_categories": dict(structure_counter),
        "matched_percentages": {
            k: (v / accepted_matches if accepted_matches else 0.0)
            for k, v in matched_counter.items()
        },
        "overall_event_percentages": {
            k: (v / total_events if total_events else 0.0)
            for k, v in {**matched_counter, **unmatched_counter}.items()
        },
        "examples": dict(examples),
    }
    return summary


def print_counter(title: str, counter_dict: dict[str, int]) -> None:
    console.print(f"\n{title}")
    if not counter_dict:
        console.print("  (none)")
        return

    for key, count in sorted(counter_dict.items(), key=lambda kv: (-kv[1], kv[0])):
        console.print(f"  {key}: {count}")


def plot_error_category_frequencies(summary: dict[str, Any], output_path: Path) -> None:
    """Plot a bar chart of all error-category frequencies."""
    category_counts: Counter[str] = Counter()
    for section in (
        "matched_categories",
        "unmatched_categories",
        "structure_categories",
    ):
        category_counts.update(summary.get(section, {}))

    if not category_counts:
        console.print("\nNo error categories to plot.")
        return

    ordered = dict(sorted(category_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    labels = list(ordered.keys())
    values = list(ordered.values())

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5.5))
    bars = ax.bar(
        [label.replace("_", " ").title() for label in labels],
        values,
        color="#4c72b0",
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(0.5, 0.005 * max(values)),
            str(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Categorize claim span errors from BIO-tagging predictions and labels "
            "using one-to-one IoU matching."
        )
    )
    parser.add_argument(
        "preds_labels_file",
        type=str,
        help="Path to JSON file with top-level 'preds' and 'labels'.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for accepting a predicted/gold match (default: 0.5).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save a JSON summary. Defaults next to input file.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=3,
        help="Maximum saved examples per error category (default: 3).",
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Do not include examples in the output JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.preds_labels_file)
    with input_path.open("r") as handle:
        payload = json.load(handle)

    rows = flatten_sequences(payload)
    summary = analyze_span_errors(
        rows=rows,
        iou_threshold=args.iou_threshold,
        max_examples_per_category=max(args.max_examples, 0),
    )

    if args.no_examples:
        summary.pop("examples", None)

    console.print("Span Error Analysis")
    console.print(f"  input: {input_path}")
    console.print(f"  turns: {summary['num_turns']}")
    console.print(f"  IoU threshold: {summary['iou_threshold']:.2f}")
    console.print(f"  gold spans: {summary['total_gold_spans']}")
    console.print(f"  predicted spans: {summary['total_pred_spans']}")
    console.print(f"  accepted matches: {summary['accepted_matches']}")

    print_counter("Matched Boundary Categories", summary["matched_categories"])
    print_counter("Unmatched Categories", summary["unmatched_categories"])
    print_counter("Structural Categories", summary["structure_categories"])

    default_output = input_path.parent / "span_error_analysis.json"
    output_json = Path(args.output_json) if args.output_json else default_output
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w") as handle:
        json.dump(summary, handle, indent=2)

    plot_path = (
        Path(str(output_json)).parent
        / "figures"
        / "span_error_category_frequencies.png"
    )
    plot_error_category_frequencies(summary, plot_path)
    console.print(f"\nSaved JSON summary to {output_json}")
    console.print(f"Saved bar chart to {plot_path}")


if __name__ == "__main__":
    main()
