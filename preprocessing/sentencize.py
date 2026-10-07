"""Convert span-annotated speaker turns into sentence-level annotations.

The input CSV must contain a text column and a spans column. ``spans`` is a
JSON list of dictionaries with half-open character offsets (``start`` and
``end``), as emitted by the annotation export script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd
from spacy.lang.en import English

try:
    from .utils import console
except ImportError:
    from utils import console


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert span annotations on speaker turns to sentence annotations."
    )
    parser.add_argument("input_csv", type=Path, help="Input span-annotated CSV")
    parser.add_argument(
        "--pooling",
        choices=("max", "mean"),
        default="max",
        help="Pool sentence token labels using max or thresholded mean (default: max)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for mean pooling; label 1 when mean is greater (default: 0.5)",
    )
    parser.add_argument(
        "--text-col", default="text", help="Text column name (default: text)"
    )
    parser.add_argument(
        "--spans-col", default="spans", help="Spans column name (default: spans)"
    )
    return parser.parse_args(argv)


def _parse_spans(value: Any) -> list[dict[str, Any]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("spans must be a JSON list")
    return [span for span in value if isinstance(span, dict)]


def _span_bounds(span: dict[str, Any]) -> Optional[tuple[int, int]]:
    try:
        start, end = int(span["start"]), int(span["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if start < 0 or end <= start:
        return None
    return start, end


def _pool_sentence(
    sentence_start: int,
    sentence_end: int,
    spans: Iterable[dict[str, Any]],
    pooling: str,
    threshold: float,
) -> tuple[int, list[dict[str, Any]]]:
    length = sentence_end - sentence_start
    labels = [0] * length
    sentence_spans: list[dict[str, Any]] = []

    for span in spans:
        bounds = _span_bounds(span)
        if bounds is None:
            continue
        span_start, span_end = bounds
        overlap_start = max(sentence_start, span_start)
        overlap_end = min(sentence_end, span_end)
        if overlap_start >= overlap_end:
            continue

        local_start = overlap_start - sentence_start
        local_end = overlap_end - sentence_start
        labels[local_start:local_end] = [1] * (local_end - local_start)
        local_span = {}
        local_span["start"] = local_start
        local_span["end"] = local_end
        local_span["text"] = span.get("text", "")
        sentence_spans.append(local_span)

    if pooling == "max":
        score = max(labels, default=0)
    else:
        mean_score = sum(labels) / length if length else 0.0
        score = int(mean_score > threshold)
    return score, sentence_spans


def sentencize(
    df: pd.DataFrame,
    pooling: str = "max",
    threshold: float = 0.5,
    text_col: str = "text",
    spans_col: str = "spans",
) -> pd.DataFrame:
    """Return one row per sentence with a pooled annotation score.

    The original turn ``id`` and ``debate_id`` columns are retained. Sentence
    numbers restart at zero for every input turn, and all other input columns
    are retained where possible.
    """
    if pooling not in {"max", "mean"}:
        raise ValueError("pooling must be 'max' or 'mean'")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    nlp = English()
    nlp.add_pipe("sentencizer")
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        text = str(row[text_col])
        spans = _parse_spans(row.get(spans_col))
        doc = nlp(text)
        for sentence_num, sentence in enumerate(doc.sents):
            score, sentence_spans = _pool_sentence(
                sentence.start_char,
                sentence.end_char,
                spans,
                pooling,
                threshold,
            )
            output_row = row.to_dict()
            output_row[text_col] = sentence.text
            output_row[spans_col] = json.dumps(sentence_spans, ensure_ascii=False)
            output_row["sentence_num"] = sentence_num
            output_row["label"] = score
            rows.append(output_row)

    columns = list(df.columns)
    for column in ("sentence_num", "label"):
        if column not in columns:
            columns.append(column)
    return pd.DataFrame(rows, columns=columns)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    dataframe = pd.read_csv(args.input_csv)
    result = sentencize(
        dataframe,
        pooling=args.pooling,
        threshold=args.threshold,
        text_col=args.text_col,
        spans_col=args.spans_col,
    )
    output_path = Path(args.input_csv).with_suffix(".sentences.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    console.print(f"Saved {len(result)} sentences to {output_path}")


if __name__ == "__main__":
    main()
