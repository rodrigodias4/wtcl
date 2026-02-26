
"""Parse a presidential debate transcript into sentence-level rows.

Input format assumptions (common in debate transcripts):
- Spoken lines typically begin with "SPEAKER: ..." (e.g., "MUIR: Good evening...").
- Metadata/comments may be prefixed by "#" and should be ignored.
- Some transcripts include C-style block comments like "/* ... */"; those are ignored.

Output:
- A CSV with one row per sentence and a single column: "sentence".
- Speaker labels are removed from the sentence text.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


SPEAKER_RE = re.compile(r"^([A-Z][A-Z .\-']{0,60}):\s*(.*)\s*$")


def _strip_inline_comment_markers(text: str) -> str:
	# Keep this conservative: remove obvious transcript artifacts.
	# Examples in the provided transcript include: [crosstalk], [laughing], [sic].
	return text.strip()


def _iter_non_comment_lines(raw_lines: Iterable[str]) -> Iterable[str]:
	"""Yield lines excluding '#' comments and C-style /* ... */ blocks."""
	in_block_comment = False

	for raw in raw_lines:
		line = raw.rstrip("\n")
		stripped = line.strip()

		if not stripped:
			continue

		if in_block_comment:
			if "*/" in stripped:
				in_block_comment = False
			continue

		if stripped.startswith("/*"):
			if "*/" not in stripped:
				in_block_comment = True
			continue

		if stripped.startswith("#"):
			continue

		yield stripped


def _collapse_whitespace(text: str) -> str:
	return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> List[str]:
	"""Split text into sentences using lightweight heuristics.

	This avoids heavy dependencies (e.g., NLTK / spaCy) while handling a few
	common abbreviations.
	"""

	text = _collapse_whitespace(text)
	if not text:
		return []

	# Protect a handful of common abbreviations / initials.
	protected = {
		"Mr.": "Mr<prd>",
		"Mrs.": "Mrs<prd>",
		"Ms.": "Ms<prd>",
		"Dr.": "Dr<prd>",
		"Prof.": "Prof<prd>",
		"Sr.": "Sr<prd>",
		"Jr.": "Jr<prd>",
		"St.": "St<prd>",
		"U.S.": "US<prd>",
		"U.K.": "UK<prd>",
		"e.g.": "eg<prd>",
		"i.e.": "ie<prd>",
		"vs.": "vs<prd>",
		"etc.": "etc<prd>",
	}
	for src, dst in protected.items():
		text = text.replace(src, dst)

	# Protect single-letter initials like "J.D." -> "JD<prd>".
	text = re.sub(r"\b([A-Z])\.([A-Z])\.", r"\1\2<prd>", text)

	# Split at sentence-ending punctuation when followed by space + a plausible next token.
	parts = re.split(r"(?<=[.!?])\s+(?=(?:\"|\'|\(|\[)?[A-Z0-9])", text)

	sentences: List[str] = []
	for part in parts:
		part = part.replace("<prd>", ".")
		part = _collapse_whitespace(part)
		if not part:
			continue
		sentences.append(part)

	return sentences


def transcript_to_sentences(transcript_path: Path) -> List[str]:
	"""Parse transcript file into a flat list of sentences (speaker removed)."""

	raw_text = transcript_path.read_text(encoding="utf-8", errors="replace")
	lines = list(_iter_non_comment_lines(raw_text.splitlines()))

	utterances: List[str] = []
	current: Optional[str] = None

	for line in lines:
		match = SPEAKER_RE.match(line)
		if match:
			spoken = match.group(2)
			spoken = _strip_inline_comment_markers(spoken)
			if current is not None:
				utterances.append(_collapse_whitespace(current))
			current = spoken
			continue

		# Continuation line: append to current utterance if any, else ignore.
		if current is None:
			continue
		current = f"{current} {line}".strip()

	if current is not None:
		utterances.append(_collapse_whitespace(current))

	sentences: List[str] = []
	for utterance in utterances:
		for sentence in split_sentences(utterance):
			sentence = _collapse_whitespace(sentence)
			if sentence:
				sentences.append(sentence)

	return sentences


def build_dataframe(sentences: List[str]) -> pd.DataFrame:
	return pd.DataFrame({"sentence": sentences})


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Parse a presidential debate transcript (.txt) into a sentence-level CSV. "
			"Lines beginning with 'SPEAKER:' have the speaker removed; '#' and /* */ comments are ignored."
		)
	)
	parser.add_argument(
		"input",
		type=Path,
		help="Path to input transcript .txt",
	)
	return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)

	if not args.input.exists():
		raise FileNotFoundError(f"Input file not found: {args.input}")

	sentences = transcript_to_sentences(args.input)
	df = build_dataframe(sentences)

	args.input.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(args.input.with_suffix(".csv"), index=False)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

