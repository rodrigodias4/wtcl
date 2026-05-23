
"""Parse all presidential debate transcripts into speaker turns.

Input format assumptions (common in debate transcripts):
- Spoken lines typically begin with "SPEAKER: ..." (e.g., "MUIR: Good evening...").
- Metadata/comments may be prefixed by "#" and should be ignored.
- Some transcripts include C-style block comments like "/* ... */"; those are ignored.
- All transcripts should be in the 'raw' folder with filenames like "YYYYMMDD_speaker1_speaker2.txt".

Output:
- A single CSV with one row per speaker turn and three columns:
  - debate_id: extracted from filename (e.g., "trump_harris")
  - speaker: the name of the speaker
  - text: the spoken text
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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


def transcript_to_speaker_turns(
	transcript_path: Path,
) -> List[Tuple[str, str]]:
	"""Parse transcript file into a list of (speaker, text) tuples.

	Each tuple represents a single speaker turn, which may span multiple
	continuation lines in the input.
	"""

	raw_text = transcript_path.read_text(encoding="utf-8", errors="replace")
	lines = list(_iter_non_comment_lines(raw_text.splitlines()))

	turns: List[Tuple[str, str]] = []
	current_speaker: Optional[str] = None
	current_text: Optional[str] = None

	for line in lines:
		match = SPEAKER_RE.match(line)
		if match:
			speaker = match.group(1)
			spoken = match.group(2)
			spoken = _strip_inline_comment_markers(spoken)

			# Save previous turn if any
			if current_speaker is not None and current_text is not None:
				text = _collapse_whitespace(current_text)
				if text:
					turns.append((current_speaker, text))

			current_speaker = speaker
			current_text = spoken
			continue

		# Continuation line: append to current turn if any, else ignore.
		if current_speaker is None:
			continue
		current_text = f"{current_text} {line}".strip()

	# Don't forget the last turn
	if current_speaker is not None and current_text is not None:
		text = _collapse_whitespace(current_text)
		if text:
			turns.append((current_speaker, text))

	return turns


def extract_debate_id(filename: str) -> str:
	"""Extract debate_id from filename.
	
	E.g., "20240910_trump_harris.txt" -> "trump_harris"
	"""
	stem = Path(filename).stem
	return stem


def parse_all_transcripts(raw_folder: Path) -> List[Tuple[str, str, str]]:
	"""Parse all .txt transcripts from raw folder into list of (debate_id, speaker, text) tuples."""
	
	all_turns: List[Tuple[str, str, str]] = []
	
	# Find all .txt files in raw folder
	transcript_files = sorted(raw_folder.glob("*.txt"))
	
	for transcript_path in transcript_files:
		debate_id = extract_debate_id(transcript_path.name)
		turns = transcript_to_speaker_turns(transcript_path)
		
		for speaker, text in turns:
			all_turns.append((debate_id, speaker, text))
	
	return all_turns


def build_dataframe(turns: List[Tuple[str, str, str]]) -> pd.DataFrame:
	return pd.DataFrame(turns, columns=["debate_id", "speaker", "text"])


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Parse all presidential debate transcripts from the 'raw' folder "
			"into a single speaker-turn CSV with debate_id, speaker, and text columns."
		)
	)
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		default=Path(__file__).parent / "all_transcripts.csv",
		help="Path to output CSV file (default: all_transcripts.csv in same folder as this script)",
	)
	return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)

	raw_folder = Path(__file__).parent / "raw"
	if not raw_folder.exists():
		raise FileNotFoundError(f"Raw folder not found: {raw_folder}")

	turns = parse_all_transcripts(raw_folder)
	df = build_dataframe(turns)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(args.output, index=False)
	print(f"Parsed {len(turns)} speaker turns from {len(df['debate_id'].unique())} debates")
	print(f"Saved to {args.output}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

