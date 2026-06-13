"""Analyze check-worthy span-annotated debate speaker turns.

This script reads either of the annotation shapes that exist in the workspace:

- Label Studio exports with a ``spans`` column containing JSON offset spans.
- Earlier CSVs with a single ``check_worthy_span`` text field.

It produces summary tables and plots for:

- speaker turn lengths
- claim span lengths
- claims per turn
- total spans and turns per debate, stacked by speaker
- claim density by normalized position inside a turn
- per-speaker turn and claim word-length ridgeline plots

Example:
	python analysis/analysis.py labelstudio/project-1-at-2026-05-22-21-39-fbc74640.csv \
		--outdir analysis_outputs/project-1
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import shutil

IGNORE_MODERATORS = True

WORD_RE = re.compile(r"\S+")

ORDER_LAST_APPEARANCE = ["GORE", "KERRY", "BUSH", "MCCAIN", "ROMNEY", "OBAMA", "CLINTON", "BIDEN", "TRUMP", "HARRIS"]

@dataclass
class SpanRecord:
    debate_id: str
    turn_index: int
    span_index: int
    speaker: str
    start: Optional[int]
    end: Optional[int]
    text: str
    reason: str
    reason_choices: List[str]
    turn_char_len: int
    turn_word_len: int
    span_char_len: Optional[int]
    span_word_len: Optional[int]
    rel_start: Optional[float]
    rel_end: Optional[float]
    rel_len: Optional[float]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze check-worthy span annotations on debate speaker turns"
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Annotated CSV to analyze",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default=None,
        help="Optional plot title prefix",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Histogram bins for length plots",
    )
    return parser.parse_args(argv)


def _first_existing_column(
    df: pd.DataFrame, candidates: Sequence[str]
) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _word_count(text: Any) -> int:
    if text is None:
        return 0
    value = str(text).strip()
    if not value:
        return 0
    return len(WORD_RE.findall(value))


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(value)
    except Exception:
        return None


def _normalize_span_text(text: Any) -> str:
    if text is None:
        return ""
    value = str(text).strip()
    if not value or value.upper() == "NULL":
        return ""
    return value


def _parse_reason_choices(raw: Any) -> List[str]:
    if raw is None:
        return []
    values: List[Any]
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, tuple):
        values = list(raw)
    else:
        text = str(raw).strip()
        if not text or text.upper() == "NULL":
            return []
        try:
            loaded = json.loads(text)
        except Exception:
            loaded = text
        if isinstance(loaded, list):
            values = loaded
        else:
            values = [loaded]

    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        choice = _normalize_span_text(value)
        if not choice:
            continue
        if choice in seen:
            continue
        seen.add(choice)
        result.append(choice)
    return result


def _parse_json_spans(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        return [raw]
    text = str(raw).strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except Exception:
        return []
    if isinstance(loaded, list):
        return [x for x in loaded if isinstance(x, dict)]
    if isinstance(loaded, dict):
        return [loaded]
    return []


def _span_word_count(text: str) -> int:
    return _word_count(text)


def _normalize_speaker(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    if isinstance(value, float) and math.isnan(value):
        return "UNKNOWN"
    text = str(value).strip()
    return text if text else "UNKNOWN"


def _normalize_debate_id(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    if isinstance(value, float) and math.isnan(value):
        return "UNKNOWN"
    text = str(value).strip()
    return text if text else "UNKNOWN"


def _normalize_id_token(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    return re.sub(r"[^A-Z0-9]+", "", text)


def _participants_from_debate_id(debate_id: Any) -> set[str]:
    if debate_id is None or (isinstance(debate_id, float) and math.isnan(debate_id)):
        return set()
    tokens = [part for part in str(debate_id).strip().split("_") if part]
    if len(tokens) <= 1:
        return set()
    # Debate IDs in this dataset start with a date token, followed by participant names.
    return {
        _normalize_id_token(token) for token in tokens[1:] if _normalize_id_token(token)
    }


def _filter_moderator_rows(df: pd.DataFrame) -> pd.DataFrame:
    if not IGNORE_MODERATORS:
        return df

    speaker_col = _first_existing_column(
        df, ["speaker", "speaker_name", "speaker_id", "participant", "author"]
    )
    if speaker_col is None or "debate_id" not in df.columns:
        return df

    keep_mask: List[bool] = []
    for _, row in df.iterrows():
        speaker = _normalize_id_token(_normalize_speaker(row.get(speaker_col)))
        participants = _participants_from_debate_id(row.get("debate_id"))
        # If debate_id is unavailable/unparseable, keep the row rather than silently dropping it.
        keep_mask.append((not participants) or (speaker in participants))

    return df.loc[keep_mask].copy()


def _extract_spans_from_row(
    row: pd.Series, text_col: str
) -> List[Tuple[Optional[int], Optional[int], str, str, List[str]]]:
    turn_text = "" if pd.isna(row.get(text_col)) else str(row.get(text_col))
    spans_col = row.get("spans") if "spans" in row.index else None
    check_span_col = (
        row.get("check_worthy_span") if "check_worthy_span" in row.index else None
    )
    results: List[Tuple[Optional[int], Optional[int], str, str, List[str]]] = []

    raw_spans = _parse_json_spans(spans_col)
    if raw_spans:
        for span in raw_spans:
            start = _safe_int(span.get("start"))
            end = _safe_int(span.get("end"))
            span_text = _normalize_span_text(span.get("text"))
            reason = _normalize_span_text(span.get("reason_text") or span.get("reason"))
            reason_choices = _parse_reason_choices(span.get("reason_choices"))
            results.append((start, end, span_text, reason, reason_choices))
        return results

    span_text = _normalize_span_text(check_span_col)
    if not span_text:
        return results

    start = turn_text.find(span_text)
    if start >= 0:
        end = start + len(span_text)
    else:
        start = None
        end = None
    results.append((start, end, span_text, "", []))
    return results


def _build_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    text_col = _first_existing_column(df, ["text", "sentence", "turn_text"])
    if text_col is None:
        raise ValueError("Expected a text column named 'text' or 'sentence'")
    speaker_col = _first_existing_column(
        df, ["speaker", "speaker_name", "speaker_id", "participant", "author"]
    )

    turn_rows: List[Dict[str, Any]] = []
    claim_rows: List[SpanRecord] = []

    for turn_index, (_, row) in enumerate(df.iterrows()):
        debate_id = _normalize_debate_id(row.get("debate_id"))
        speaker = (
            _normalize_speaker(row.get(speaker_col))
            if speaker_col is not None
            else "ALL"
        )
        turn_text = "" if pd.isna(row.get(text_col)) else str(row.get(text_col))
        turn_char_len = len(turn_text)
        turn_word_len = _word_count(turn_text)
        turn_spans = _extract_spans_from_row(row, text_col=text_col)

        rel_positions: List[float] = []
        rel_lengths: List[float] = []
        span_char_lengths: List[int] = []
        span_word_lengths: List[int] = []

        for span_index, (start, end, span_text, reason, reason_choices) in enumerate(
            turn_spans
        ):
            span_char_len: Optional[int]
            span_word_len: Optional[int]
            rel_start: Optional[float]
            rel_end: Optional[float]
            rel_len: Optional[float]

            if start is not None and end is not None and end >= start:
                span_char_len = end - start
                span_word_len = (
                    _span_word_count(turn_text[start:end]) if turn_text else None
                )
                if turn_char_len > 0:
                    rel_start = start / turn_char_len
                    rel_end = end / turn_char_len
                    rel_len = span_char_len / turn_char_len
                else:
                    rel_start = None
                    rel_end = None
                    rel_len = None
            else:
                span_char_len = len(span_text) if span_text else None
                span_word_len = _span_word_count(span_text) if span_text else None
                rel_start = None
                rel_end = None
                rel_len = None

            if rel_start is not None:
                rel_positions.append(rel_start)
            if rel_len is not None:
                rel_lengths.append(rel_len)
            if span_char_len is not None:
                span_char_lengths.append(span_char_len)
            if span_word_len is not None:
                span_word_lengths.append(span_word_len)

            claim_rows.append(
                SpanRecord(
                    debate_id=debate_id,
                    turn_index=turn_index,
                    span_index=span_index,
                    speaker=speaker,
                    start=start,
                    end=end,
                    text=span_text,
                    reason=reason,
                    reason_choices=reason_choices,
                    turn_char_len=turn_char_len,
                    turn_word_len=turn_word_len,
                    span_char_len=span_char_len,
                    span_word_len=span_word_len,
                    rel_start=rel_start,
                    rel_end=rel_end,
                    rel_len=rel_len,
                )
            )

        turn_rows.append(
            {
                "debate_id": debate_id,
                "turn_index": turn_index,
                "speaker": speaker,
                "turn_char_len": turn_char_len,
                "turn_word_len": turn_word_len,
                "claim_count": len(turn_spans),
                "claim_chars_total": sum(span_char_lengths),
                "claim_words_total": sum(span_word_lengths),
                "mean_rel_start": (
                    sum(rel_positions) / len(rel_positions) if rel_positions else None
                ),
                "mean_rel_len": (
                    sum(rel_lengths) / len(rel_lengths) if rel_lengths else None
                ),
            }
        )

    turn_metrics = pd.DataFrame(turn_rows)
    claim_metrics = pd.DataFrame([asdict(row) for row in claim_rows])
    return turn_metrics, claim_metrics


def _compute_speaker_metrics(
    turn_metrics: pd.DataFrame, claim_metrics: pd.DataFrame
) -> pd.DataFrame:
    if turn_metrics.empty or "speaker" not in turn_metrics.columns:
        return pd.DataFrame(
            columns=[
                "speaker",
                "turn_count",
                "turns_with_claims",
                "turns_with_claims_pct",
                "claim_count",
                "spans_per_turn",
                "turn_chars_total",
                "claim_chars_total",
                "span_chars_over_turn_chars_pct",
                "turn_words_total",
                "claim_words_total",
                "span_words_over_turn_words_pct",
            ]
        )

    base = turn_metrics.copy()
    base["speaker"] = base["speaker"].fillna("UNKNOWN").astype(str)

    grouped = (
        base.groupby("speaker", dropna=False)
        .agg(
            turn_count=("turn_index", "count"),
            turns_with_claims=("claim_count", lambda s: int((s > 0).sum())),
            claim_count=("claim_count", "sum"),
            turn_chars_total=("turn_char_len", "sum"),
            claim_chars_total=("claim_chars_total", "sum"),
            turn_words_total=("turn_word_len", "sum"),
            claim_words_total=("claim_words_total", "sum"),
        )
        .reset_index()
    )

    grouped["turns_with_claims_pct"] = np.where(
        grouped["turn_count"] > 0,
        100.0 * grouped["turns_with_claims"] / grouped["turn_count"],
        0.0,
    )
    grouped["spans_per_turn"] = np.where(
        grouped["turn_count"] > 0,
        grouped["claim_count"] / grouped["turn_count"],
        0.0,
    )
    grouped["span_chars_over_turn_chars_pct"] = np.where(
        grouped["turn_chars_total"] > 0,
        100.0 * grouped["claim_chars_total"] / grouped["turn_chars_total"],
        0.0,
    )
    grouped["span_words_over_turn_words_pct"] = np.where(
        grouped["turn_words_total"] > 0,
        100.0 * grouped["claim_words_total"] / grouped["turn_words_total"],
        0.0,
    )

    return grouped.sort_values(
        ["spans_per_turn", "claim_count"], ascending=[False, False]
    ).reset_index(drop=True)


def _compute_reason_choice_metrics(claim_metrics: pd.DataFrame) -> pd.DataFrame:
    if claim_metrics.empty or "speaker" not in claim_metrics.columns:
        return pd.DataFrame(columns=["reason_choice", "speaker", "count", "total_count"])

    records: List[Dict[str, Any]] = []
    for _, row in claim_metrics.iterrows():
        speaker = _normalize_speaker(row.get("speaker"))
        for choice in _parse_reason_choices(row.get("reason_choices")):
            records.append({"reason_choice": choice, "speaker": speaker})

    if not records:
        return pd.DataFrame(columns=["reason_choice", "speaker", "count", "total_count"])

    choice_counts = (
        pd.DataFrame(records)
        .groupby(["reason_choice", "speaker"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    totals = (
        choice_counts.groupby("reason_choice", dropna=False)["count"]
        .sum()
        .reset_index(name="total_count")
    )
    merged = choice_counts.merge(totals, on="reason_choice", how="left")
    return merged.sort_values(
        ["total_count", "reason_choice", "count"], ascending=[False, True, False]
    ).reset_index(drop=True)


def _save_summary(
    turn_metrics: pd.DataFrame,
    claim_metrics: pd.DataFrame,
    speaker_metrics: pd.DataFrame,
    reason_choice_metrics: pd.DataFrame,
    outdir: Path,
) -> None:
    per_speaker: Dict[str, Dict[str, Any]] = {}
    if not speaker_metrics.empty:
        for _, row in speaker_metrics.iterrows():
            speaker = str(row["speaker"])
            per_speaker[speaker] = {
                "turn_count": int(row["turn_count"]),
                "claim_count": int(row["claim_count"]),
                "spans_per_turn": float(row["spans_per_turn"]),
                "span_chars_over_turn_chars_pct": float(
                    row["span_chars_over_turn_chars_pct"]
                ),
                "span_words_over_turn_words_pct": float(
                    row["span_words_over_turn_words_pct"]
                ),
            }

    summary = {
        "turn_count": int(len(turn_metrics)),
        "claim_count": int(len(claim_metrics)),
        "speaker_count": (
            int(speaker_metrics["speaker"].nunique())
            if not speaker_metrics.empty
            else 0
        ),
        "turns_with_claims": (
            int((turn_metrics["claim_count"] > 0).sum())
            if not turn_metrics.empty
            else 0
        ),
        "turn_length_chars": _series_summary(
            turn_metrics["turn_char_len"]
            if not turn_metrics.empty
            else pd.Series(dtype=float)
        ),
        "turn_length_words": _series_summary(
            turn_metrics["turn_word_len"]
            if not turn_metrics.empty
            else pd.Series(dtype=float)
        ),
        "claims_per_turn": _series_summary(
            turn_metrics["claim_count"]
            if not turn_metrics.empty
            else pd.Series(dtype=float)
        ),
        "claim_span_chars": _series_summary(
            claim_metrics["span_char_len"].dropna()
            if not claim_metrics.empty
            else pd.Series(dtype=float)
        ),
        "claim_span_words": _series_summary(
            claim_metrics["span_word_len"].dropna()
            if not claim_metrics.empty
            else pd.Series(dtype=float)
        ),
        "claim_rel_start": _series_summary(
            claim_metrics["rel_start"].dropna()
            if not claim_metrics.empty
            else pd.Series(dtype=float)
        ),
        "claim_rel_len": _series_summary(
            claim_metrics["rel_len"].dropna()
            if not claim_metrics.empty
            else pd.Series(dtype=float)
        ),
        "reason_choice_count": (
            int(reason_choice_metrics["count"].sum())
            if not reason_choice_metrics.empty
            else 0
        ),
        "reason_choice_totals": (
            reason_choice_metrics.groupby("reason_choice", dropna=False)["count"]
            .sum()
            .sort_values(ascending=False)
            .to_dict()
            if not reason_choice_metrics.empty
            else {}
        ),
        "per_speaker": per_speaker,
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _series_summary(series: pd.Series) -> Dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0}
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=0)) if clean.count() > 1 else 0.0,
        "min": float(clean.min()),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
        "max": float(clean.max()),
    }


def _figure_path(outdir: Path, filename: str) -> Path:
    return outdir / filename


def _debate_year_label(debate_id: Any) -> str:
    text = _normalize_debate_id(debate_id)
    match = re.search(r"(19|20)\d{2}", text)
    return match.group(0) if match else text


def _decorate_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)


def _set_reasonable_xticks(
    ax: plt.Axes, left: float, right: float, max_ticks: int = 6
) -> None:
    if not math.isfinite(left) or not math.isfinite(right):
        return
    if left == right:
        ax.set_xticks([left])
        return
    span = right - left
    step = max(span / max_ticks, 1)
    if span <= 20:
        step = 1
    else:
        step = max(1, int(round(step / 5)) * 5)
    start = math.floor(left / step) * step
    stop = math.ceil(right / step) * step
    ax.set_xticks(np.arange(start, stop + step, step))


def _hist_plot(
    series: pd.Series,
    path: Path,
    title: str,
    xlabel: str,
    bins: int,
    color: str,
    log_y: bool = False,
    align_integer_bins: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        if align_integer_bins:
            left = math.floor(float(clean.min())) - 0.5
            right = math.ceil(float(clean.max())) + 0.5
            bin_edges = np.arange(left, right + 1.0, 1.0)
        else:
            bin_edges = np.histogram_bin_edges(clean, bins=bins)
        ax.hist(
            clean, bins=bin_edges, color=color, edgecolor="white", alpha=0.9, log=log_y
        )
        # Use integer y-axis ticks for histogram counts
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        _decorate_axis(ax, title, xlabel, "Count")
        ax.set_xlim(bin_edges[0], bin_edges[-1])
        if align_integer_bins:
            start = int(math.ceil(bin_edges[0] + 0.5))
            stop = int(math.floor(bin_edges[-1] - 0.5))
            if start <= stop:
                ax.set_xticks(np.arange(start, stop + 1, 1))
            else:
                ax.set_xticks([round(float(clean.min()))])
        else:
            _set_reasonable_xticks(ax, float(bin_edges[0]), float(bin_edges[-1]))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_turn_lengths(
    turn_metrics: pd.DataFrame, outdir: Path, bins: int, title_prefix: str
) -> None:
    _hist_plot(
        turn_metrics["turn_word_len"],
        _figure_path(outdir, "turn_lengths_words.png"),
        f"{title_prefix}Turn Lengths" if title_prefix else "Turn Lengths",
        "Words per turn",
        bins,
        "#264653",
    )

    _hist_plot(
        turn_metrics["turn_char_len"],
        _figure_path(outdir, "turn_lengths_chars.png"),
        (
            f"{title_prefix}Turn Lengths (Characters)"
            if title_prefix
            else "Turn Lengths (Characters)"
        ),
        "Characters per turn",
        bins,
        "#2a9d8f",
    )


def _plot_claim_span_lengths(
    claim_metrics: pd.DataFrame, outdir: Path, bins: int, title_prefix: str
) -> None:
    _hist_plot(
        claim_metrics["span_word_len"],
        _figure_path(outdir, "claim_span_lengths_words.png"),
        f"{title_prefix}Claim Span Lengths" if title_prefix else "Claim Span Lengths",
        "Words per claim span",
        bins,
        "#e76f51",
    )

    _hist_plot(
        claim_metrics["span_char_len"],
        _figure_path(outdir, "claim_span_lengths_chars.png"),
        (
            f"{title_prefix}Claim Span Lengths (Characters)"
            if title_prefix
            else "Claim Span Lengths (Characters)"
        ),
        "Characters per claim span",
        bins,
        "#f4a261",
    )


def _plot_claims_per_turn(
    turn_metrics: pd.DataFrame, outdir: Path, bins: int, title_prefix: str
) -> None:
    _hist_plot(
        turn_metrics["claim_count"],
        _figure_path(outdir, "claims_per_turn.png"),
        f"{title_prefix}Claims per Turn" if title_prefix else "Claims per Turn",
        "Claims in a turn",
        (
            min(bins, max(10, int(turn_metrics["claim_count"].max()) + 1))
            if not turn_metrics.empty
            else bins
        ),
        "#6d597a",
        align_integer_bins=True,
    )


def _plot_debate_totals_by_speaker(
    turn_metrics: pd.DataFrame,
    claim_metrics: pd.DataFrame,
    outdir: Path,
    title_prefix: str,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))

    has_turn_debates = not turn_metrics.empty and "debate_id" in turn_metrics.columns
    has_claim_debates = not claim_metrics.empty and "debate_id" in claim_metrics.columns
    if not has_turn_debates and not has_claim_debates:
        ax.text(
            0.5,
            0.5,
            "No debate_id data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(_figure_path(outdir, "debate_total_spans_turns.png"), dpi=300)
        plt.close(fig)
        return

    turn_source = (
        turn_metrics.dropna(subset=["debate_id", "speaker"])
        if has_turn_debates
        else pd.DataFrame(columns=["debate_id", "speaker"])
    )
    claim_source = (
        claim_metrics.dropna(subset=["debate_id", "speaker"])
        if has_claim_debates
        else pd.DataFrame(columns=["debate_id", "speaker"])
    )

    if turn_source.empty and claim_source.empty:
        ax.text(
            0.5,
            0.5,
            "No debate_id data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(_figure_path(outdir, "debate_total_spans_turns.png"), dpi=300)
        plt.close(fig)
        return

    turn_pivot = (
        turn_source.pivot_table(
            index="debate_id",
            columns="speaker",
            values="turn_index",
            aggfunc="count",
            fill_value=0,
        )
        if not turn_source.empty
        else pd.DataFrame()
    )
    span_pivot = (
        claim_source.pivot_table(
            index="debate_id",
            columns="speaker",
            values="span_index",
            aggfunc="count",
            fill_value=0,
        )
        if not claim_source.empty
        else pd.DataFrame()
    )

    debates = sorted(set(turn_pivot.index) | set(span_pivot.index))
    if not debates:
        ax.text(
            0.5,
            0.5,
            "No debate_id data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(_figure_path(outdir, "debate_total_spans_turns.png"), dpi=300)
        plt.close(fig)
        return

    turn_pivot = turn_pivot.reindex(debates, fill_value=0)
    span_pivot = span_pivot.reindex(debates, fill_value=0)
    speaker_order = (
        pd.concat([
            turn_pivot.sum(axis=0),
            span_pivot.sum(axis=0),
        ], axis=1)
        .fillna(0)
        .sum(axis=1)
        .sort_values(ascending=False)
        .index.tolist()
    )

    x = np.arange(len(debates))
    width = 0.36
    speaker_cmap = plt.get_cmap("tab20")
    speaker_colors = {
        speaker: speaker_cmap(
            float(idx) / float(max(1, len(speaker_order) - 1))
        )
        if len(speaker_order) > 1
        else speaker_cmap(0.5)
        for idx, speaker in enumerate(speaker_order)
    }

    for offset, pivot in [
        (-width / 2, turn_pivot),
        (width / 2, span_pivot),
    ]:
        bottoms = np.zeros(len(debates), dtype=float)
        for speaker in speaker_order:
            values = (
                pivot[speaker].to_numpy(dtype=float)
                if speaker in pivot.columns
                else np.zeros(len(debates), dtype=float)
            )
            if np.all(values == 0):
                continue
            ax.bar(
                x + offset,
                values,
                width,
                bottom=bottoms,
                color=speaker_colors[speaker],
                edgecolor="white",
                linewidth=0.6,
                alpha=0.95,
            )
            bottoms += values

        for idx, total in enumerate(bottoms):
            if total > 0:
                ax.text(
                    x[idx] + offset,
                    total + max(0.1, total * 0.015),
                    f"{int(total)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    _decorate_axis(
        ax,
        (
            f"{title_prefix}Total Spans and Turns per Debate by Speaker"
            if title_prefix
            else "Total Spans and Turns per Debate by Speaker"
        ),
        "Debate",
        "Count",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_debate_year_label(debate) for debate in debates], rotation=30, ha="right"
    )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    max_total = 0.0
    if not turn_pivot.empty:
        max_total = max(max_total, float(turn_pivot.sum(axis=1).max()))
    if not span_pivot.empty:
        max_total = max(max_total, float(span_pivot.sum(axis=1).max()))
    if max_total > 0:
        ax.set_ylim(0, max_total * 1.12)

    speaker_handles = [
        Patch(facecolor=speaker_colors[speaker], edgecolor="white", label=str(speaker))
        for speaker in speaker_order
    ]
    if speaker_handles:
        ax.legend(
            handles=speaker_handles,
            title="Speaker",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
        )

    fig.text(
        0.08,
        0.92,
        "Left bar = Turns, Right bar = Spans",
        ha="left",
        va="top",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(_figure_path(outdir, "debate_total_spans_turns.png"), dpi=300)
    plt.close(fig)


def _plot_claim_density(
    claim_metrics: pd.DataFrame, outdir: Path, title_prefix: str
) -> None:
    clean = (
        claim_metrics.dropna(subset=["rel_start", "rel_len"])
        if not claim_metrics.empty
        else claim_metrics
    )

    def _plot_single_claim_density(
        path: Path,
        title: str,
        xlabel: str,
        ylabel: str,
        plot_fn: Any,
        use_colorbar: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        if clean.empty:
            ax.text(
                0.5,
                0.5,
                "No offset spans available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
        else:
            result = plot_fn(ax)
            _decorate_axis(ax, title, xlabel, ylabel)
            ax.set_xlim(0.0, 1.0)
            if use_colorbar and result is not None:
                ax.set_ylim(0.0, 1.0)
                fig.colorbar(result[3], ax=ax, label="Claims")
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)

    def _plot_span_occupancy_histogram(path: Path, title: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        if clean.empty:
            ax.text(
                0.5,
                0.5,
                "No offset spans available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
        else:
            bins = np.linspace(0.0, 1.0, 41)
            occupancy = np.zeros(len(bins) - 1, dtype=float)
            for _, row in clean.iterrows():
                start = row.get("rel_start")
                end = row.get("rel_end")
                if pd.isna(start) or pd.isna(end):
                    continue
                left = float(min(start, end))
                right = float(max(start, end))
                if right <= 0.0 or left >= 1.0:
                    continue
                left = max(0.0, left)
                right = min(1.0, right)
                start_idx = max(0, int(np.searchsorted(bins, left, side="right") - 1))
                end_idx = min(len(occupancy) - 1, int(np.searchsorted(bins, right, side="left")))
                occupancy[start_idx : end_idx + 1] += 1.0

            x = bins[:-1] + np.diff(bins) / 2.0
            ax.bar(
                bins[:-1],
                occupancy,
                width=np.diff(bins),
                align="edge",
                color="#2a9d8f",
                edgecolor="white",
                alpha=0.82,
            )

            finite_mask = np.isfinite(x) & np.isfinite(occupancy)
            if finite_mask.sum() >= 3:
                centered_x = x[finite_mask] - 0.5
                centered_y = occupancy[finite_mask]
                max_degree = min(5, len(centered_x) - 1)
                if max_degree >= 1:
                    coeffs = np.polyfit(centered_x, centered_y, deg=max_degree)
                    approximation = np.poly1d(coeffs)(x - 0.5)
                    approximation = np.clip(approximation, 0.0, None)
                    ax.plot(
                        x,
                        approximation,
                        color="#e76f51",
                        linewidth=2.2,
                        linestyle="--",
                        label=f"Taylor approximation (deg {max_degree})",
                    )

            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            _decorate_axis(
                ax,
                title,
                "Normalized position in turn",
                "Span Occurrences",
            )
            ax.set_xlim(0.0, 1.0)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)

    _plot_single_claim_density(
        _figure_path(outdir, "claim_start_location.png"),
        f"{title_prefix}Claim Start Location" if title_prefix else "Claim Start Location",
        "Relative start position in turn",
        "Claims",
        lambda ax: ax.hist(
            clean["rel_start"],
            bins=40,
            range=(0.0, 1.0),
            color="#457b9d",
            edgecolor="white",
        ),
    )

    _plot_single_claim_density(
        _figure_path(outdir, "claim_relative_span_size.png"),
        (
            f"{title_prefix}Claim Relative Span Size"
            if title_prefix
            else "Claim Relative Span Size"
        ),
        "Span length / turn length",
        "Claims",
        lambda ax: ax.hist(
            clean["rel_len"],
            bins=40,
            range=(0.0, 1.0),
            color="#8e44ad",
            edgecolor="white",
        ),
    )

    _plot_span_occupancy_histogram(
        _figure_path(outdir, "claim_span_occupancy.png"),
        (
            f"{title_prefix}Claim Span Occupancy"
            if title_prefix
            else "Claim Span Occupancy"
        ),
    )

    _plot_single_claim_density(
        _figure_path(outdir, "claim_start_vs_relative_span_size.png"),
        (
            f"{title_prefix}Claim Start vs Relative Span Size"
            if title_prefix
            else "Claim Start vs Relative Span Size"
        ),
        "Relative start position in turn",
        "Span length / turn length",
        lambda ax: ax.hist2d(
            clean["rel_start"],
            clean["rel_len"],
            bins=40,
            range=[[0.0, 1.0], [0.0, 1.0]],
            cmap="BuPu",
        ),
        use_colorbar=True,
    )


def _plot_speaker_spans_per_turn(
    speaker_metrics: pd.DataFrame, outdir: Path, title_prefix: str
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    if speaker_metrics.empty:
        ax.text(
            0.5,
            0.5,
            "No speaker data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        df = speaker_metrics.sort_values("spans_per_turn", ascending=False)
        speakers = df["speaker"].tolist()
        x = np.arange(len(speakers))
        n_speakers = len(speakers)
        colormap = plt.get_cmap("PuBu_r")
        colormap = [colormap(float(idx) / float(n_speakers + 4)) if n_speakers > 1 else colormap(0.5) for idx in range(n_speakers)]
        ax.bar(x, df["spans_per_turn"], color=colormap, alpha=0.9)
        _decorate_axis(
            ax,
            (
                f"{title_prefix}Spans Per Turn by Speaker"
                if title_prefix
                else "Spans Per Turn by Speaker"
            ),
            "Speaker",
            "Spans / turn",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(speakers, rotation=30, ha="right")
        ax.yaxis.set_major_locator(MaxNLocator(integer=False))

    fig.tight_layout()
    fig.savefig(_figure_path(outdir, "speaker_spans_per_turn.png"), dpi=300)
    plt.close(fig)


def _plot_speaker_span_coverage(
    speaker_metrics: pd.DataFrame, outdir: Path, title_prefix: str
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    if speaker_metrics.empty:
        ax.text(
            0.5,
            0.5,
            "No speaker data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        df = speaker_metrics.sort_values(
            "span_chars_over_turn_chars_pct", ascending=False
        )
        speakers = df["speaker"].tolist()
        x = np.arange(len(speakers))
        width = 0.4
        ax.bar(
            x - width / 2,
            df["span_chars_over_turn_chars_pct"],
            width,
            label="Chars",
            color="#20578b",
            alpha=0.9,
        )
        ax.bar(
            x + width / 2,
            df["span_words_over_turn_words_pct"],
            width,
            label="Words",
            color="#6a96d2",
            alpha=0.9,
        )
        _decorate_axis(
            ax,
            (
                f"{title_prefix}Span Coverage by Speaker"
                if title_prefix
                else "Span Coverage by Speaker"
            ),
            "Speaker",
            "Span length / turn length (%)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(speakers, rotation=30, ha="right")
        ax.legend()

    fig.tight_layout()
    fig.savefig(_figure_path(outdir, "speaker_span_coverage.png"), dpi=300)
    plt.close(fig)


def _plot_speaker_span_count(
    speaker_metrics: pd.DataFrame, outdir: Path, title_prefix: str
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    if speaker_metrics.empty:
        ax.text(
            0.5,
            0.5,
            "No speaker data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(_figure_path(outdir, "speaker_span_counts.png"), dpi=300)
        plt.close(fig)
        return

    # Use a consistent ordering: sort by claim_count descending
    df = speaker_metrics.sort_values("claim_count", ascending=False)
    speakers = df["speaker"].tolist()
    x = np.arange(len(speakers))

    width = 0.35
    bars_spans = ax.bar(x - width / 2, df["claim_count"], width, color="#8F6FA3", alpha=0.9, label="Total spans")
    bars_turns = ax.bar(x + width / 2, df["turn_count"], width, color="#2B223F", alpha=0.9, label="Total turns")
    ax.set_xticks(x)
    ax.set_xticklabels(speakers, rotation=30, ha="right")
    _decorate_axis(
        ax,
        (
            f"{title_prefix}Total Check-worthy Spans and Turns by Speaker"
            if title_prefix
            else "Total Check-worthy Spans and Turns by Speaker"
        ),
        "Speaker",
        "Count",
    )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Legend for both bar sets
    handles = [bars_spans, bars_turns]
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(_figure_path(outdir, "speaker_total_counts.png"), dpi=300)
    plt.close(fig)


def _plot_speaker_word_lengths(
    turn_metrics: pd.DataFrame,
    claim_metrics: pd.DataFrame,
    outdir: Path,
    bins: int,
    title_prefix: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def _gaussian_smooth(values: np.ndarray, sigma_bins: float = 1.3) -> np.ndarray:
        if len(values) == 0:
            return values
        radius = max(1, int(math.ceil(sigma_bins * 3)))
        offsets = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
        kernel /= kernel.sum()
        return np.convolve(values, kernel, mode="same")

    def _plot_ridgeline(
        ax: plt.Axes,
        data_by_speaker: Dict[str, pd.Series],
        subtitle: str,
        speaker_order: List[str],
    ) -> None:
        clean_by_speaker = {
            speaker: pd.to_numeric(series, errors="coerce").dropna()
            for speaker, series in data_by_speaker.items()
        }
        combined = pd.concat(
            [series for series in clean_by_speaker.values() if not series.empty],
            ignore_index=True,
        ) if any(not series.empty for series in clean_by_speaker.values()) else pd.Series(dtype=float)

        if combined.empty:
            ax.text(
                0.5,
                0.5,
                "No speaker data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return

        x_min = float(combined.min())
        x_max = float(combined.max())
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5

        x_grid = np.linspace(x_min, x_max, 400)
        bin_edges = np.histogram_bin_edges(combined, bins=bins)
        colors = plt.get_cmap("tab10")
        max_density = 0.0
        ridge_data: List[Tuple[str, np.ndarray]] = []

        for speaker in speaker_order:
            series = clean_by_speaker.get(speaker, pd.Series(dtype=float))
            if series.empty:
                ridge_data.append((speaker, np.zeros_like(x_grid)))
                continue
            hist, edges = np.histogram(series.to_numpy(), bins=bin_edges, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            density = np.interp(x_grid, centers, hist, left=0.0, right=0.0)
            density = _gaussian_smooth(density)
            max_density = max(max_density, float(density.max(initial=0.0)))
            ridge_data.append((speaker, density))

        if max_density <= 0:
            ax.text(
                0.5,
                0.5,
                "No speaker data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return

        row_height = 1.0
        fill_scale = 0.85
        for idx, (speaker, density) in enumerate(ridge_data):
            offset = idx * row_height
            scaled = (density / max_density) * fill_scale
            color = colors(idx % 10)
            ax.fill_between(x_grid, offset, offset + scaled, color=color, alpha=0.75)
            ax.plot(x_grid, offset + scaled, color=color, linewidth=1.2)
            ax.hlines(offset, x_min, x_max, color="#444444", linewidth=0.5, alpha=0.35)

        ax.set_yticks(np.arange(len(speaker_order)) * row_height)
        ax.set_yticklabels(speaker_order)
        ax.set_ylim(-0.4, max(0.6, (len(speaker_order) - 1) * row_height + 1.0))
        ax.set_xlim(x_min, x_max)
        _decorate_axis(ax, subtitle, "Words", "Speaker")
        _set_reasonable_xticks(ax, x_min, x_max)

    turn_data = (
        turn_metrics.groupby("speaker", dropna=False)["turn_word_len"].apply(list)
        if not turn_metrics.empty and "speaker" in turn_metrics.columns
        else pd.Series(dtype=object)
    )
    claim_data = (
        claim_metrics.groupby("speaker", dropna=False)["span_word_len"].apply(list)
        if not claim_metrics.empty and "speaker" in claim_metrics.columns
        else pd.Series(dtype=object)
    )

    turn_series_by_speaker = {
        str(speaker): pd.Series(values)
        for speaker, values in turn_data.items()
    }
    claim_series_by_speaker = {
        str(speaker): pd.Series(values)
        for speaker, values in claim_data.items()
    }

    order_source = {
        speaker: pd.to_numeric(series, errors="coerce").dropna().median()
        for speaker, series in turn_series_by_speaker.items()
        if not pd.to_numeric(series, errors="coerce").dropna().empty
    }
    speaker_order = [speaker for speaker, _ in sorted(order_source.items(), key=lambda item: item[1])]
    if not speaker_order:
        speaker_order = sorted(set(turn_series_by_speaker) | set(claim_series_by_speaker))

    _plot_ridgeline(
        axes[0],
        turn_series_by_speaker,
        f"{title_prefix}Turn Word Lengths by Speaker"
        if title_prefix
        else "Turn Word Lengths by Speaker",
        ORDER_LAST_APPEARANCE,
    )
    _plot_ridgeline(
        axes[1],
        claim_series_by_speaker,
        f"{title_prefix}Span Word Lengths by Speaker"
        if title_prefix
        else "Span Word Lengths by Speaker",
        ORDER_LAST_APPEARANCE,
    )

    fig.tight_layout()
    fig.savefig(_figure_path(outdir, "speaker_word_lengths.png"), dpi=300)
    plt.close(fig)


def _plot_speaker_metrics(
    turn_metrics: pd.DataFrame,
    claim_metrics: pd.DataFrame,
    speaker_metrics: pd.DataFrame,
    outdir: Path,
    bins: int,
    title_prefix: str,
) -> None:
    legacy_path = _figure_path(outdir, "speaker_metrics.png")
    if legacy_path.exists():
        legacy_path.unlink()
    _plot_speaker_spans_per_turn(speaker_metrics, outdir, title_prefix)
    _plot_speaker_span_coverage(speaker_metrics, outdir, title_prefix)
    _plot_speaker_span_count(speaker_metrics, outdir, title_prefix)
    _plot_speaker_word_lengths(turn_metrics, claim_metrics, outdir, bins=bins, title_prefix=title_prefix)


def _plot_reason_choice_rankings(
    reason_choice_metrics: pd.DataFrame, outdir: Path, title_prefix: str
) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))

    if reason_choice_metrics.empty:
        ax.text(
            0.5,
            0.5,
            "No reason choices available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(_figure_path(outdir, "reason_choice_rankings.png"), dpi=300)
        plt.close(fig)
        return

    pivot = (
        reason_choice_metrics.pivot_table(
            index="reason_choice",
            columns="speaker",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .astype(float)
    )
    totals = pivot.sum(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[totals.index]
    speaker_order = pivot.sum(axis=0).sort_values(ascending=False).index.tolist()

    y = np.arange(len(pivot.index))
    left = np.zeros(len(pivot.index), dtype=float)
    cmap = plt.get_cmap("rainbow")
    n_speakers = len(speaker_order)

    for idx, speaker in enumerate(speaker_order):
        values = pivot[speaker].to_numpy(dtype=float)
        if np.all(values == 0):
            continue
        ax.barh(
            y,
            values,
            left=left,
            color=cmap(float(idx) / float(n_speakers - 1)) if n_speakers > 1 else cmap(0.5),
            alpha=0.9,
            label=str(speaker),
        )
        left += values

    for idx, total in enumerate(totals.to_numpy(dtype=float)):
        ax.text(total + 0.15, idx, f"{int(total)}", va="center", ha="left", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index.tolist())
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _decorate_axis(
        ax,
        (
            f"{title_prefix}Reason Choice Frequency by Speaker"
            if title_prefix
            else "Reason Choice Frequency by Speaker"
        ),
        "Labeled spans",
        "Reason choice",
    )
    ax.set_xlim(0, max(1.0, float(left.max()) * 1.12))

    if speaker_order:
        legend_cols = 2 if len(speaker_order) > 8 else 1
        ax.legend(title="Speaker", ncol=legend_cols, loc="lower right")

    fig.tight_layout()
    fig.savefig(_figure_path(outdir, "reason_choice_rankings.png"), dpi=300)
    plt.close(fig)


def _plot_reason_choice_correlation(
    claim_metrics: pd.DataFrame, outdir: Path, title_prefix: str
) -> None:
    # Expect `reason_choices` column to contain lists of choices per claim
    if claim_metrics.empty or "reason_choices" not in claim_metrics.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(
            0.5,
            0.5,
            "No reason choice data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(_figure_path(outdir, "reason_choice_correlation.png"), dpi=300)
        plt.close(fig)
        return

    # Build binary matrix of shape (n_claims, n_reason_choices)
    records = []
    all_choices: List[str] = []
    for _, row in claim_metrics.iterrows():
        raw = row.get("reason_choices")
        # normalize to list
        choices = []
        if isinstance(raw, list):
            choices = [str(x) for x in raw if x is not None]
        elif pd.isna(raw):
            choices = []
        else:
            try:
                # maybe a JSON string
                parsed = json.loads(str(raw))
                if isinstance(parsed, list):
                    choices = [str(x) for x in parsed if x is not None]
                else:
                    choices = [str(parsed)]
            except Exception:
                choices = [str(raw)] if raw else []

        normalized = [_normalize_span_text(c) for c in choices]
        normalized = [c for c in normalized if c]
        records.append(normalized)
        for c in normalized:
            if c not in all_choices:
                all_choices.append(c)

    if not all_choices or len(all_choices) < 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(
            0.5,
            0.5,
            "Not enough distinct reason choices for correlation",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(_figure_path(outdir, "reason_choice_correlation.png"), dpi=300)
        plt.close(fig)
        return

    # Create DataFrame
    bin_rows: List[Dict[str, int]] = []
    for choices in records:
        row = {choice: (1 if choice in choices else 0) for choice in all_choices}
        bin_rows.append(row)

    bin_df = pd.DataFrame(bin_rows, columns=all_choices)
    if bin_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(
            0.5,
            0.5,
            "No reason choice data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(_figure_path(outdir, "reason_choice_correlation.png"), dpi=300)
        plt.close(fig)
        return

    # Compute Pearson correlation matrix (suitable for binary co-occurrence / phi)
    corr = bin_df.corr()

    size = max(10.0, 0.4 * len(all_choices) + 4.0)
    fig, ax = plt.subplots(figsize=(size, size))

    im = ax.imshow(corr.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(all_choices)))
    ax.set_yticks(np.arange(len(all_choices)))
    ax.set_xticklabels(all_choices, rotation=45, ha="right")
    ax.set_yticklabels(all_choices)
    #ax.tick_params(axis="both", labelsize=9)
    ax.set_title(
        (f"{title_prefix}Reason Choice Correlation" if title_prefix else "Reason Choice Correlation")
    )

    # annotate
    for i in range(len(all_choices)):
        for j in range(len(all_choices)):
            val = corr.iat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    fig.tight_layout()
    fig.savefig(_figure_path(outdir, "reason_choice_correlation.png"), dpi=300)
    plt.close(fig)



def _write_tables(
    turn_metrics: pd.DataFrame,
    claim_metrics: pd.DataFrame,
    speaker_metrics: pd.DataFrame,
    reason_choice_metrics: pd.DataFrame,
    outdir: Path,
) -> None:
    turn_metrics.to_csv(outdir / "turn_metrics.csv", index=False)
    claim_metrics.to_csv(outdir / "claim_metrics.csv", index=False)
    speaker_metrics.to_csv(outdir / "speaker_metrics.csv", index=False)
    reason_choice_metrics.to_csv(outdir / "reason_choice_metrics.csv", index=False)


def analyze(
    input_csv: Path, outdir: Path, bins: int, title_prefix: Optional[str]
) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input not found: {input_csv}")

    outdir.mkdir(parents=True, exist_ok=True)
    # Clear existing output directory to avoid stale files from previous runs
    if outdir.exists():
        for child in outdir.iterdir():
            try:
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
            except Exception:
                # If we can't remove a file, continue; user can inspect manually
                pass

    df = pd.read_csv(input_csv)
    df = _filter_moderator_rows(df)
    turn_metrics, claim_metrics = _build_metrics(df)
    speaker_metrics = _compute_speaker_metrics(turn_metrics, claim_metrics)
    reason_choice_metrics = _compute_reason_choice_metrics(claim_metrics)

    _write_tables(
        turn_metrics,
        claim_metrics,
        speaker_metrics,
        reason_choice_metrics,
        outdir,
    )
    _save_summary(
        turn_metrics,
        claim_metrics,
        speaker_metrics,
        reason_choice_metrics,
        outdir,
    )

    prefix = f"{title_prefix} - " if title_prefix else ""
    _plot_turn_lengths(turn_metrics, outdir, bins=bins, title_prefix=prefix)
    _plot_claim_span_lengths(claim_metrics, outdir, bins=bins, title_prefix=prefix)
    _plot_claims_per_turn(turn_metrics, outdir, bins=bins, title_prefix=prefix)
    _plot_debate_totals_by_speaker(turn_metrics, claim_metrics, outdir, title_prefix=prefix)
    _plot_claim_density(claim_metrics, outdir, title_prefix=prefix)
    _plot_speaker_metrics(
        turn_metrics,
        claim_metrics,
        speaker_metrics,
        outdir,
        bins=bins,
        title_prefix=prefix,
    )
    _plot_reason_choice_rankings(reason_choice_metrics, outdir, title_prefix=prefix)
    _plot_reason_choice_correlation(claim_metrics, outdir, title_prefix=prefix)

    print(f"Analyzed {len(turn_metrics)} turns and {len(claim_metrics)} claims")
    if not speaker_metrics.empty:
        print("Per-speaker check-worthy metrics:")
        for _, row in speaker_metrics.iterrows():
            print(
                f"  {row['speaker']}: "
                f"spans/turn={row['spans_per_turn']:.3f}, "
                f"span-chars/turn-chars={row['span_chars_over_turn_chars_pct']:.2f}%, "
                f"span-words/turn-words={row['span_words_over_turn_words_pct']:.2f}%"
            )
    print(f"Wrote outputs to {outdir}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    outdir = Path(__file__).resolve().parent / "output"
    

    analyze(
        args.input_csv, outdir=outdir, bins=args.bins, title_prefix=args.title_prefix
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
