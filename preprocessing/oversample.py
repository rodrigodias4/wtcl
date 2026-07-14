"""Weighted oversampling for debate datasets.

Sampling weights are based on two factors:
1) Claim span density (rows with denser claim spans get higher weights)
2) Debate totals (rows from debates with many rows get lower weights)
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
            "Oversample a CSV using span density and inverse debate-size weighting."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Input CSV path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input>.oversampled.csv)",
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
        "--target-size",
        type=int,
        default=None,
        help=(
            "Rows to sample with replacement. "
            "Default balances debate totals as max_debate_rows * num_debates."
        ),
    )
    parser.add_argument(
        "--density-scale",
        type=float,
        default=2.0,
        help="Scale for density effect in weight = (1 + density_scale * density)",
    )
    parser.add_argument(
        "--debate-power",
        type=float,
        default=1.0,
        help="Power for inverse debate weight: (1 / debate_count)^debate_power",
    )
    parser.add_argument(
        "--density-correction-power",
        "--density-balance-power",
        dest="density_balance_power",
        type=float,
        default=1.0,
        help=(
            "Strength of the density correction toward the mean debate span density. "
            "Higher values push debate densities back toward the corpus mean more aggressively."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--keep-weight-cols",
        action="store_true",
        help="Keep helper weight columns in output",
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


def add_sampling_weights(
    df: pd.DataFrame,
    debate_col: str,
    text_col: str,
    spans_col: str,
    density_scale: float,
    debate_power: float,
    density_balance_power: float,
) -> pd.DataFrame:
    weighted = df.copy()
    weighted["_span_density"] = weighted.apply(
        lambda row: compute_span_density(row[text_col], row[spans_col]), axis=1
    )

    debate_density_means = weighted.groupby(debate_col)["_span_density"].mean()
    target_debate_density = float(debate_density_means.mean())
    balance_denominator = debate_density_means + 1e-9
    debate_density_balance = ((target_debate_density + 1e-9) / balance_denominator).pow(
        density_balance_power
    )
    weighted["_debate_density_mean"] = weighted[debate_col].map(debate_density_means)
    weighted["_density_balance"] = weighted[debate_col].map(debate_density_balance)

    weighted["_debate_count"] = weighted.groupby(debate_col)[debate_col].transform(
        "count"
    )

    weighted["_density_weight"] = (
        1.0 + (density_scale * weighted["_span_density"])
    ) * weighted["_density_balance"]
    weighted["_debate_weight"] = (1.0 / weighted["_debate_count"]).pow(debate_power)
    weighted["_sample_weight"] = (
        weighted["_density_weight"] * weighted["_debate_weight"]
    )

    total = weighted["_sample_weight"].sum()
    if total <= 0:
        raise ValueError("Computed non-positive total sample weight.")
    weighted["_sample_weight"] = weighted["_sample_weight"] / total

    return weighted


def add_source_row_id(df: pd.DataFrame) -> pd.DataFrame:
    weighted = df.copy()
    weighted["_source_row_id"] = weighted.index
    return weighted


def resolve_target_size(
    df: pd.DataFrame, debate_col: str, explicit: Optional[int]
) -> int:
    if explicit is not None:
        if explicit <= 0:
            raise ValueError("--target-size must be a positive integer")
        return max(explicit, len(df))

    debate_counts = df.groupby(debate_col).size()
    if debate_counts.empty:
        raise ValueError("Input dataframe is empty")
    return max(int(debate_counts.max() * debate_counts.shape[0]), len(df))


def weighted_oversample(df: pd.DataFrame, target_size: int, seed: int) -> pd.DataFrame:
    if target_size < len(df):
        raise ValueError(
            "target_size must be at least the number of input rows to keep every row once"
        )

    remaining = target_size - len(df)
    if remaining == 0:
        return df.copy().reset_index(drop=True)

    sampled = df.sample(
        n=remaining,
        replace=True,
        weights="_sample_weight",
        random_state=seed,
    )

    return pd.concat([df, sampled], ignore_index=True)


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def print_debug_stats(
    original_df: pd.DataFrame,
    oversampled_df: pd.DataFrame,
    debate_col: str,
) -> None:
    row_counts = oversampled_df["_source_row_id"].value_counts().sort_index()

    if row_counts.empty:
        console.print("No rows were sampled.")
        return

    duplication_stats = row_counts.describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    duplicated_rows = int((row_counts > 1).sum())
    total_duplicates = int((row_counts - 1).clip(lower=0).sum())

    console.print("Oversample debug stats:")
    console.print(f"  input rows: {len(original_df)}")
    console.print(f"  output rows: {len(oversampled_df)}")
    console.print(f"  unique source rows selected: {len(row_counts)}")
    console.print(f"  rows selected more than once: {duplicated_rows}")
    console.print(f"  total duplicate draws: {total_duplicates}")
    console.print(
        "  duplicate count per source row: "
        f"min={_format_float(float(duplication_stats['min']))}, "
        f"median={_format_float(float(duplication_stats['50%']))}, "
        f"mean={_format_float(float(duplication_stats['mean']))}, "
        f"p90={_format_float(float(duplication_stats['90%']))}, "
        f"p95={_format_float(float(duplication_stats['95%']))}, "
        f"p99={_format_float(float(duplication_stats['99%']))}, "
        f"max={_format_float(float(duplication_stats['max']))}"
    )

    top_rows = row_counts.sort_values(ascending=False).head(5)
    console.print(
        "  most duplicated source rows: "
        + ", ".join(f"{int(row_id)}x{int(count)}" for row_id, count in top_rows.items())
    )

    debate_counts = (
        oversampled_df.groupby(debate_col).size().sort_values(ascending=False)
    )
    debate_stats = debate_counts.describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    console.print(
        "  sampled rows per debate: "
        f"min={_format_float(float(debate_stats['min']))}, "
        f"median={_format_float(float(debate_stats['50%']))}, "
        f"mean={_format_float(float(debate_stats['mean']))}, "
        f"p90={_format_float(float(debate_stats['90%']))}, "
        f"p95={_format_float(float(debate_stats['95%']))}, "
        f"p99={_format_float(float(debate_stats['99%']))}, "
        f"max={_format_float(float(debate_stats['max']))}"
    )

    original_debate_density = original_df.copy()
    if "_span_density" in oversampled_df.columns:
        original_debate_density["_span_density"] = original_df.apply(
            lambda row: compute_span_density(row.get("text"), row.get("spans")),
            axis=1,
        )
        original_density_means = original_debate_density.groupby(debate_col)[
            "_span_density"
        ].mean()
        sampled_density_means = oversampled_df.groupby(debate_col)[
            "_span_density"
        ].mean()
        density_summary = original_density_means.describe(
            percentiles=[0.5, 0.9, 0.95, 0.99]
        )
        sampled_density_summary = sampled_density_means.describe(
            percentiles=[0.5, 0.9, 0.95, 0.99]
        )
        original_mean = float(original_density_means.mean())
        sampled_mean = float(sampled_density_means.mean())
        console.print(
            "  debate mean span density before: "
            f"min={_format_float(float(density_summary['min']))}, "
            f"median={_format_float(float(density_summary['50%']))}, "
            f"mean={_format_float(original_mean)}, "
            f"p90={_format_float(float(density_summary['90%']))}, "
            f"p95={_format_float(float(density_summary['95%']))}, "
            f"p99={_format_float(float(density_summary['99%']))}, "
            f"max={_format_float(float(density_summary['max']))}"
        )
        console.print(
            "  debate mean span density after: "
            f"min={_format_float(float(sampled_density_summary['min']))}, "
            f"median={_format_float(float(sampled_density_summary['50%']))}, "
            f"mean={_format_float(sampled_mean)}, "
            f"p90={_format_float(float(sampled_density_summary['90%']))}, "
            f"p95={_format_float(float(sampled_density_summary['95%']))}, "
            f"p99={_format_float(float(sampled_density_summary['99%']))}, "
            f"max={_format_float(float(sampled_density_summary['max']))}"
        )
        console.print(
            f"  mean debate density shift: {_format_float(sampled_mean - original_mean)}"
        )


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing!r}. Available: {list(df.columns)!r}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input not found: {args.input_csv}")

    output = args.output
    if output is None:
        output = args.input_csv.with_suffix(".oversampled.csv")

    df = pd.read_csv(args.input_csv)
    validate_columns(df, [args.debate_col, args.text_col, args.spans_col])
    console.print(f"Loaded {len(df)} rows from {args.input_csv}")

    weighted = add_sampling_weights(
        df=df,
        debate_col=args.debate_col,
        text_col=args.text_col,
        spans_col=args.spans_col,
        density_scale=args.density_scale,
        debate_power=args.debate_power,
        density_balance_power=args.density_balance_power,
    )
    weighted = add_source_row_id(weighted)

    target_size = resolve_target_size(weighted, args.debate_col, args.target_size)
    console.print(f"Sampling target size: {target_size}")
    console.print(f"Density correction power: {args.density_balance_power}")

    oversampled = weighted_oversample(weighted, target_size=target_size, seed=args.seed)
    print_debug_stats(df, oversampled, debate_col=args.debate_col)

    if not args.keep_weight_cols:
        oversampled = oversampled.drop(
            columns=[
                "_span_density",
                "_debate_count",
                "_density_weight",
                "_debate_weight",
                "_debate_density_mean",
                "_density_balance",
                "_sample_weight",
                "_source_row_id",
            ]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    oversampled.sort_values(by=["id"], inplace=True, ignore_index=True)
    oversampled.to_csv(output, index=False)

    console.print(f"Wrote {len(oversampled)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
