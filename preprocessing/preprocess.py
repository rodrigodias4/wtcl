"""Filter debate turn CSVs before span extraction.

This script removes:
- moderator turns, defined as speakers not present in the debate_id token list
- turns with fewer than 4 words

It reports how many rows are removed at each stage and writes the filtered CSV.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd


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
		raise KeyError(f"Missing columns: {missing!r}. Available columns: {list(df.columns)!r}")


def filter_turns(
	df: pd.DataFrame,
	debate_col: str = "debate_id",
	speaker_col: str = "speaker",
	text_col: str = "text",
	min_words: int = 4,
) -> tuple[pd.DataFrame, int, int]:
	"""Remove moderator turns and short turns.

	Returns the filtered dataframe plus the number of rows removed at each stage.
	"""
	_validate_columns(df, [debate_col, speaker_col, text_col])

	stage1_removed = 0
	keep_stage1 = []
	for _, row in df.iterrows():
		participants = _debate_participants(row.get(debate_col))
		speaker = _speaker_token(row.get(speaker_col))
		# Keep rows when the debate id cannot be parsed; only drop clear moderator rows.
		keep = (not participants) or (speaker in participants)
		keep_stage1.append(keep)
		if not keep:
			stage1_removed += 1

	filtered = df.loc[keep_stage1].copy()

	stage2_mask = filtered[text_col].map(lambda value: _word_count(value) >= min_words)
	stage2_removed = int((~stage2_mask).sum())
	filtered = filtered.loc[stage2_mask].copy()

	return filtered, stage1_removed, stage2_removed


def main(argv: Optional[Sequence[str]] = None) -> int:
	args = parse_args(argv)
	if not args.input_csv.exists():
		raise FileNotFoundError(f"Input not found: {args.input_csv}")

	out = args.output
	if out is None:
		out = args.input_csv.with_suffix(".filtered.csv")

	df = pd.read_csv(args.input_csv)
	filtered, moderator_removed, short_removed = filter_turns(
		df,
		debate_col=args.debate_col,
		speaker_col=args.speaker_col,
		text_col=args.text_col,
		min_words=args.min_words,
	)

	out.parent.mkdir(parents=True, exist_ok=True)
	filtered.to_csv(out, index=False)

	print(f"Loaded {len(df)} rows from {args.input_csv}")
	print(f"Removed {moderator_removed} moderator turns")
	print(f"Removed {short_removed} turns with fewer than {args.min_words} words")
	print(f"Kept {len(filtered)} rows")
	print(f"Wrote {out}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
