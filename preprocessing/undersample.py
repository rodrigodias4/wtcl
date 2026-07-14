"""Weighted undersampling for debate datasets.

This script reduces only the largest outlier debate by default. Rows kept from
oversized debates are selected in two stages: keep rows with spans first, then
fill remaining slots with longer rows, using deterministic ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from utils import console


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Undersample a CSV by reducing the largest debates first, while "
            "favoring rows with higher span density and longer text."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Input CSV path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input>.undersampled.csv)",
    )
    parser.add_argument(
        "--debate-col",
        type=str,
        default="debate_id",
        help="Debate id column (default: debate_id)",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="text",
        help="Text column (default: text)",
    )
    parser.add_argument(
        "--spans-col",
        type=str,
        default="spans",
        help="Spans column storing JSON span arrays (default: spans)",
    )
    parser.add_argument(
        "--max-rows-per-debate",
        type=int,
        default=None,
        help=(
            "Maximum rows to keep per debate. Default: the size of the second-"
            "largest debate in the input, so only the largest outlier debate is reduced."
        ),
    )
    parser.add_argument(
        "--span-density-scale",
        "--density-scale",
        dest="span_density_scale",
        type=float,
        default=2.0,
        help=(
            "Strength of the span-density preference when trimming a debate "
            "that contains more span rows than the cap."
        ),
    )
    parser.add_argument(
        "--length-scale",
        type=float,
        default=1.0,
        help=(
            "Strength of the length preference in the keep score: "
            "(1 + length_scale * normalized_length)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Kept for CLI compatibility; selection is deterministic.",
    )
    parser.add_argument(
        "--keep-weight-cols",
        action="store_true",
        help="Keep helper score columns in output",
    )
    return parser.parse_args(argv)


def _parse_spans(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, list):
        return [span for span in value if isinstance(span, dict)]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [span for span in parsed if isinstance(span, dict)]
    return []


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def compute_span_density(text: Any, spans_value: Any) -> float:
    text_str = (
        "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text)
    )
    text_len = max(1, len(text_str))

    spans = _parse_spans(spans_value)
    intervals: list[tuple[int, int]] = []
    for span in spans:
        start = span.get("start")
        end = span.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if end <= start:
            continue
        clamped_start = max(0, min(start, text_len))
        clamped_end = max(0, min(end, text_len))
        if clamped_end > clamped_start:
            intervals.append((clamped_start, clamped_end))

    merged = _merge_intervals(intervals)
    covered_chars = sum(end - start for start, end in merged)
    return covered_chars / text_len


def compute_text_length(text: Any) -> float:
    if text is None:
        return 0.0
    if isinstance(text, float) and pd.isna(text):
        return 0.0
    return float(len(str(text)))


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing!r}. Available: {list(df.columns)!r}"
        )


def add_keep_scores(
    df: pd.DataFrame,
    debate_col: str,
    text_col: str,
    spans_col: str,
    span_density_scale: float,
    length_scale: float,
) -> pd.DataFrame:
    weighted = df.copy()
    weighted["_span_density"] = weighted.apply(
        lambda row: compute_span_density(row[text_col], row[spans_col]), axis=1
    )
    weighted["_text_length"] = weighted[text_col].map(compute_text_length)
    weighted["_has_spans"] = weighted["_span_density"] > 0

    max_length = float(weighted["_text_length"].max())
    if max_length <= 0:
        weighted["_length_norm"] = 0.0
    else:
        weighted["_length_norm"] = weighted["_text_length"] / max_length

    weighted["_span_keep_score"] = (
        1.0 + span_density_scale * weighted["_span_density"]
    ) * (1.0 + length_scale * weighted["_length_norm"])
    weighted["_length_keep_score"] = 1.0 + length_scale * weighted["_length_norm"]
    weighted["_debate_count"] = weighted.groupby(debate_col)[debate_col].transform(
        "count"
    )
    return weighted


def resolve_max_rows_per_debate(
    df: pd.DataFrame, debate_col: str, explicit: Optional[int]
) -> int:
    if explicit is not None:
        if explicit <= 0:
            raise ValueError("--max-rows-per-debate must be a positive integer")
        return explicit

    debate_counts = df.groupby(debate_col).size()
    if debate_counts.empty:
        raise ValueError("Input dataframe is empty")

    sorted_counts = sorted(debate_counts.tolist(), reverse=True)
    if len(sorted_counts) == 1:
        return int(sorted_counts[0])

    return int(sorted_counts[1])


def weighted_undersample_debates(
    df: pd.DataFrame,
    debate_col: str,
    max_rows_per_debate: int,
    seed: int,
) -> pd.DataFrame:
    kept_groups: list[pd.DataFrame] = []
    removed_rows = 0

    for index, (debate, group) in enumerate(
        sorted(df.groupby(debate_col), key=lambda item: str(item[0]))
    ):
        if len(group) <= max_rows_per_debate:
            kept_groups.append(group)
            continue

        span_rows = group[group["_has_spans"]]
        non_span_rows = group[~group["_has_spans"]]

        if len(span_rows) >= max_rows_per_debate:
            sampled = span_rows.sort_values(
                by=["_span_density", "_text_length", "id"],
                ascending=[False, False, True],
                kind="mergesort",
            ).head(max_rows_per_debate)
        else:
            remaining = max_rows_per_debate - len(span_rows)
            if remaining > 0 and len(non_span_rows) > 0:
                non_span_sample = non_span_rows.sort_values(
                    by=["_text_length", "id"],
                    ascending=[False, True],
                    kind="mergesort",
                ).head(min(remaining, len(non_span_rows)))
                sampled = pd.concat([span_rows, non_span_sample], ignore_index=False)
            else:
                sampled = span_rows.copy()

        sampled = sampled.sort_values(by=["id"], kind="mergesort")
        removed_rows += len(group) - len(sampled)
        kept_groups.append(sampled)

    if not kept_groups:
        return df.iloc[0:0].copy()

    result = pd.concat(kept_groups, ignore_index=False)
    result.attrs["removed_rows"] = removed_rows
    return result


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def print_debug_stats(
    original_df: pd.DataFrame,
    undersampled_df: pd.DataFrame,
    debate_col: str,
) -> None:
    original_counts = (
        original_df.groupby(debate_col).size().sort_values(ascending=False)
    )
    sampled_counts = (
        undersampled_df.groupby(debate_col).size().sort_values(ascending=False)
    )

    original_density = original_df.apply(
        lambda row: compute_span_density(row.get("text"), row.get("spans")), axis=1
    )
    sampled_density = undersampled_df.apply(
        lambda row: compute_span_density(row.get("text"), row.get("spans")), axis=1
    )
    original_length = original_df["text"].map(compute_text_length)
    sampled_length = undersampled_df["text"].map(compute_text_length)

    console.print("Undersample debug stats:")
    console.print(f"  input rows: {len(original_df)}")
    console.print(f"  output rows: {len(undersampled_df)}")
    console.print(f"  rows removed: {len(original_df) - len(undersampled_df)}")
    console.print(
        f"  debates reduced: {int((sampled_counts < original_counts.reindex(sampled_counts.index, fill_value=0)).sum())}"
    )

    top_original = original_counts.head(5)
    top_sampled = sampled_counts.head(5)
    console.print(
        "  largest debates before: "
        + ", ".join(f"{debate}={int(count)}" for debate, count in top_original.items())
    )
    console.print(
        "  largest debates after: "
        + ", ".join(f"{debate}={int(count)}" for debate, count in top_sampled.items())
    )

    original_density_stats = original_density.describe(
        percentiles=[0.5, 0.9, 0.95, 0.99]
    )
    sampled_density_stats = sampled_density.describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    original_length_stats = original_length.describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    sampled_length_stats = sampled_length.describe(percentiles=[0.5, 0.9, 0.95, 0.99])

    console.print(
        "  span density before: "
        f"min={_format_float(float(original_density_stats['min']))}, "
        f"median={_format_float(float(original_density_stats['50%']))}, "
        f"mean={_format_float(float(original_density_stats['mean']))}, "
        f"p90={_format_float(float(original_density_stats['90%']))}, "
        f"p95={_format_float(float(original_density_stats['95%']))}, "
        f"p99={_format_float(float(original_density_stats['99%']))}, "
        f"max={_format_float(float(original_density_stats['max']))}"
    )
    console.print(
        "  span density after: "
        f"min={_format_float(float(sampled_density_stats['min']))}, "
        f"median={_format_float(float(sampled_density_stats['50%']))}, "
        f"mean={_format_float(float(sampled_density_stats['mean']))}, "
        f"p90={_format_float(float(sampled_density_stats['90%']))}, "
        f"p95={_format_float(float(sampled_density_stats['95%']))}, "
        f"p99={_format_float(float(sampled_density_stats['99%']))}, "
        f"max={_format_float(float(sampled_density_stats['max']))}"
    )
    console.print(
        "  text length before: "
        f"min={_format_float(float(original_length_stats['min']))}, "
        f"median={_format_float(float(original_length_stats['50%']))}, "
        f"mean={_format_float(float(original_length_stats['mean']))}, "
        f"p90={_format_float(float(original_length_stats['90%']))}, "
        f"p95={_format_float(float(original_length_stats['95%']))}, "
        f"p99={_format_float(float(original_length_stats['99%']))}, "
        f"max={_format_float(float(original_length_stats['max']))}"
    )
    console.print(
        "  text length after: "
        f"min={_format_float(float(sampled_length_stats['min']))}, "
        f"median={_format_float(float(sampled_length_stats['50%']))}, "
        f"mean={_format_float(float(sampled_length_stats['mean']))}, "
        f"p90={_format_float(float(sampled_length_stats['90%']))}, "
        f"p95={_format_float(float(sampled_length_stats['95%']))}, "
        f"p99={_format_float(float(sampled_length_stats['99%']))}, "
        f"max={_format_float(float(sampled_length_stats['max']))}"
    )
    console.print(
        f"  rows with spans kept: {int((undersampled_df.apply(lambda row: compute_span_density(row.get('text'), row.get('spans')) > 0, axis=1)).sum())}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input not found: {args.input_csv}")

    output = args.output
    if output is None:
        output = args.input_csv.with_suffix(".undersampled.csv")

    df = pd.read_csv(args.input_csv)
    validate_columns(df, [args.debate_col, args.text_col, args.spans_col])
    console.print(f"Loaded {len(df)} rows from {args.input_csv}")

    max_rows_per_debate = resolve_max_rows_per_debate(
        df, args.debate_col, args.max_rows_per_debate
    )
    console.print(f"Max rows per debate: {max_rows_per_debate}")
    console.print(f"Span density scale: {args.span_density_scale}")
    console.print(f"Length scale: {args.length_scale}")

    weighted = add_keep_scores(
        df=df,
        debate_col=args.debate_col,
        text_col=args.text_col,
        spans_col=args.spans_col,
        span_density_scale=args.span_density_scale,
        length_scale=args.length_scale,
    )

    undersampled = weighted_undersample_debates(
        weighted,
        debate_col=args.debate_col,
        max_rows_per_debate=max_rows_per_debate,
        seed=args.seed,
    )
    print_debug_stats(df, undersampled, debate_col=args.debate_col)

    if not args.keep_weight_cols:
        undersampled = undersampled.drop(
            columns=[
                "_span_density",
                "_text_length",
                "_length_norm",
                "_has_spans",
                "_debate_count",
            ]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    undersampled.sort_values(by=["id"], inplace=True, ignore_index=True)
    undersampled.to_csv(output, index=False)

    console.print(f"Wrote {len(undersampled)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
