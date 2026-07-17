"""Filter debate turn CSVs before span extraction.

This script removes:
- moderator turns, defined as speakers not present in the debate_id token list
- turns with fewer than 4 words

It reports how many rows are removed at each stage and writes the filtered CSV.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from utils import console

WORD_RE = re.compile(r"\S+")


def _normalize_token(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return re.sub(r"[^A-Z0-9]+", "", str(value).strip().upper())


def _debate_participants(debate_id: Any) -> set[str]:
    if debate_id is None or (isinstance(debate_id, float) and math.isnan(debate_id)):
        return set()
    tokens = [token for token in str(debate_id).strip().split("_") if token]
    if len(tokens) <= 1:
        return set()
    # Debate ids in this corpus start with a date token, followed by the participants.
    return {_normalize_token(token) for token in tokens[1:] if _normalize_token(token)}


def _speaker_token(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return _normalize_token(value)


def _word_count(text: Any) -> int:
    if text is None:
        return 0
    value = str(text).strip()
    if not value:
        return 0
    return len(WORD_RE.findall(value))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove moderator turns and short turns from a speaker-turn CSV."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input CSV with debate_id, speaker, and text columns",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input>.filtered.csv)",
    )
    parser.add_argument(
        "--speaker-col",
        type=str,
        default="speaker",
        help="Speaker column name (default: speaker)",
    )
    parser.add_argument(
        "--debate-col",
        type=str,
        default="debate_id",
        help="Debate id column name (default: debate_id)",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="text",
        help="Text column name (default: text)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=4,
        help="Minimum number of words required to keep a turn (default: 4)",
    )
    return parser.parse_args(argv)


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns: {missing!r}. Available columns: {list(df.columns)!r}"
        )


def moderator_turns(
    df: pd.DataFrame,
    debate_col: str = "debate_id",
    speaker_col: str = "speaker",
    text_col: str = "text",
    min_words: int = 4,
) -> tuple[pd.DataFrame, int]:
    """Remove moderator turns and short turns.

    Returns the filtered dataframe plus the number of rows removed at each stage.
    """
    _validate_columns(df, [debate_col, speaker_col, text_col])

    removed = 0
    keep_stage1 = []
    for _, row in df.iterrows():
        participants = _debate_participants(row.get(debate_col))
        speaker = _speaker_token(row.get(speaker_col))
        # Keep rows when the debate id cannot be parsed; only drop clear moderator rows.
        keep = (not participants) or (speaker in participants)
        keep_stage1.append(keep)
        if not keep:
            removed += 1

    filtered = df.loc[keep_stage1].copy()

    return filtered, removed


def min_words(
    df: pd.DataFrame,
    debate_col: str = "debate_id",
    speaker_col: str = "speaker",
    text_col: str = "text",
    min_words: int = 4,
) -> tuple[pd.DataFrame, int]:
    """Filter turns by minimum word count."""
    df_min_words = df.copy()
    for i, row in df.iterrows():
        if (
            _word_count(row[text_col]) < min_words
            and row["spans"] == "[]"
            and row[text_col].endswith("-")
        ):
            console.print(
                f"Removing row {row['id']} from {row[debate_col]} with text: {row[text_col]!r} (word count: {_word_count(row[text_col])}) (spans: {row['spans']})"
            )
            df_min_words = df_min_words.drop(i)
    return df_min_words, len(df) - len(df_min_words)


def remove_speech_markers(
    df: pd.DataFrame,
    text_col: str,
    leading_sequences: Sequence[str] = ("--", "..."),
    trailing_sequences: Sequence[str] = ("--"),
) -> pd.DataFrame:
    def regex_replace_with_span_map(text, pattern, replacement):
        """
        Returns:
        cleaned_text
        map: original_index -> cleaned_index (or -1 if deleted)
        """

        matches = list(re.finditer(pattern, text))

        orig_to_clean = [-1] * len(text)
        cleaned = []
        j = 0
        i = 0

        while i < len(text):
            if matches and i == matches[0].start():
                match = matches.pop(0)
                cleaned.append(replacement)
                anchor = j + (replacement.find("—") if "—" in replacement else 0)
                for k in range(match.start(), match.end()):
                    orig_to_clean[k] = anchor
                i = match.end()
                j += len(replacement)
                continue

            orig_to_clean[i] = j
            cleaned.append(text[i])
            i += 1
            j += 1

        return "".join(cleaned), orig_to_clean

    def regex_clean_with_span_map(text, pattern):
        """
        Returns:
        cleaned_text
        map: original_index -> cleaned_index (or -1 if deleted)
        """

        matches = list(re.finditer(pattern, text))

        remove_ranges = [(m.start(), m.end()) for m in matches]

        orig_to_clean = [-1] * len(text)

        cleaned = []
        j = 0
        i = 0

        remove_i = 0

        while i < len(text):

            # skip removal spans
            if remove_i < len(remove_ranges) and i == remove_ranges[remove_i][0]:
                i = remove_ranges[remove_i][1]
                remove_i += 1
                continue

            orig_to_clean[i] = j
            cleaned.append(text[i])

            i += 1
            j += 1

        return "".join(cleaned), orig_to_clean

    def strip_boundary_sequences_with_span_map(text, leading, trailing):
        start = 0
        end = len(text)

        while True:
            changed = False

            while start < end and text[start].isspace():
                start += 1
                changed = True

            while start < end:
                matched = False
                for sequence in leading:
                    if sequence and text.startswith(sequence, start, end):
                        start += len(sequence)
                        matched = True
                        changed = True
                        break
                if not matched:
                    break

            while end > start and text[end - 1].isspace():
                end -= 1
                changed = True

            while end > start:
                matched = False
                for sequence in trailing:
                    if sequence and text.endswith(sequence, start, end):
                        end -= len(sequence)
                        matched = True
                        changed = True
                        break
                if not matched:
                    break

            if not changed:
                break

        orig_to_clean = [-1] * len(text)
        cleaned = []

        for i in range(start, end):
            orig_to_clean[i] = len(cleaned)
            cleaned.append(text[i])

        return "".join(cleaned), orig_to_clean

    def remap_span(span, mapping, text):
        new = [
            mapping[i] for i in range(span["start"], span["end"]) if mapping[i] != -1
        ]

        if not new:
            return None

        return {
            "start": min(new),
            "end": max(new) + 1,
            "text": text[min(new) : max(new) + 1],
        }

    df_cleaned = df.copy()
    for i, row in df.iterrows():
        text = row[text_col]
        spans = json.loads(row["spans"])
        # Replace fancy apostrophes with standard apostrophes
        text = re.sub("’", "'", text)
        # Remove configured boundary markers before interior normalization.
        text, orig_to_clean = strip_boundary_sequences_with_span_map(
            text, leading_sequences, trailing_sequences
        )
        spans = [remap_span(span, orig_to_clean, text) for span in spans]
        # Normalize speaker pauses / self-repairs into a spaced em dash while keeping offsets anchored.
        text, orig_to_clean = regex_replace_with_span_map(text, r"--", " — ")
        spans = [remap_span(span, orig_to_clean, text) for span in spans]
        # Remove content in square brackets
        text, orig_to_clean = regex_clean_with_span_map(text, r"\[.*\]")
        spans = [remap_span(span, orig_to_clean, text) for span in spans]
        # Remove content in parentheses
        text, orig_to_clean = regex_clean_with_span_map(text, r"\(.*\)")
        spans = [remap_span(span, orig_to_clean, text) for span in spans]
        # Replace multiple spaces with a single space
        text, orig_to_clean = regex_replace_with_span_map(text, r"\s\s+", " ")
        spans = [remap_span(span, orig_to_clean, text) for span in spans]
        # Trim remaining edge whitespace after cleanup.
        text, orig_to_clean = strip_boundary_sequences_with_span_map(text, (), ())
        spans = [remap_span(span, orig_to_clean, text) for span in spans]
        df_cleaned.loc[i, text_col] = text
        df_cleaned.loc[i, "spans"] = json.dumps(spans, ensure_ascii=False)
    return df_cleaned


def clip_turns_per_debate(
    df: pd.DataFrame,
    debate_col: str = "debate_id",
    max_turns_per_debate: Optional[int] = None,
) -> tuple[pd.DataFrame, int]:
    """Clip turns per debate."""
    clipped_removed = 0
    if max_turns_per_debate is None:
        return df, 0

    # Group by debate and clip turns
    def clip_group(group):
        if len(group) <= max_turns_per_debate:
            return group

        nonlocal clipped_removed
        clipped_removed += len(group) - max_turns_per_debate
        return group.sample(n=min(len(group), max_turns_per_debate), random_state=42)

    return (
        df.groupby(debate_col, as_index=True).apply(clip_group).reset_index(level=0),
        clipped_removed,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input not found: {args.input_csv}")

    out = args.output
    if out is None:
        out = args.input_csv.with_suffix(".filtered.csv")

    df = pd.read_csv(args.input_csv)
    console.print(f"Loaded {len(df)} rows from {args.input_csv}")

    preprocessed, removed = moderator_turns(
        df,
        debate_col=args.debate_col,
        speaker_col=args.speaker_col,
        text_col=args.text_col,
    )
    console.print(f"Removed {removed} moderator turns")

    preprocessed, removed = min_words(
        preprocessed,
        debate_col=args.debate_col,
        speaker_col=args.speaker_col,
        text_col=args.text_col,
        min_words=args.min_words,
    )
    console.print(f"Removed {removed} turns with fewer than {args.min_words} words")

    preprocessed = remove_speech_markers(preprocessed, text_col=args.text_col)

    # Drop empty text rows after filtering
    preprocessed = preprocessed[preprocessed[args.text_col].notnull()]
    preprocessed = preprocessed[preprocessed[args.text_col] != ""]

    # Drop rows without alphanumeric characters
    preprocessed = preprocessed[
        preprocessed[args.text_col].str.contains(r"[a-zA-Z0-9]", na=False)
    ]

    """ preprocessed, removed = clip_turns_per_debate(
        filtered,
        debate_col=args.debate_col,
        max_turns_per_debate=200,
    )
    console.print(f"Clipped {removed} rows")"""

    out.parent.mkdir(parents=True, exist_ok=True)
    preprocessed.sort_values("id").to_csv(out, index=False)

    console.print(f"Kept {len(preprocessed)} rows")
    console.print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
