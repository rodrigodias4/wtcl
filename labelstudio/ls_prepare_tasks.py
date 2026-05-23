"""Convert a speaker-turn CSV into Label Studio JSON tasks.

This script handles the output from `parse_transcript.py`, which produces CSVs with:
  debate_id, speaker, text

Label Studio expects a list of tasks, where each task has a `data` dict.

We preserve the debate_id and speaker in task data so exports can be mapped back to the original debate.

Example:
  python labelstudio/ls_prepare_tasks.py transcripts/all_transcripts.csv \
    --out labelstudio/tasks_all_debates.json

Then in Label Studio: Import -> JSON -> select the generated tasks file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Prepare Label Studio tasks JSON from speaker-turn CSV")
	p.add_argument("input_csv", type=Path, help="Input CSV path (with debate_id, speaker, text columns)")
	p.add_argument(
		"--out",
		type=Path,
		default=None,
		help="Output JSON path (default: <input>.labelstudio.json)",
	)
	p.add_argument(
		"--text-col",
		type=str,
		default="text",
		help="Column that contains the text to annotate (default: text)",
	)
	p.add_argument(
		"--debate-col",
		type=str,
		default="debate_id",
		help="Column for debate identifier (default: debate_id)",
	)
	p.add_argument(
		"--speaker-col",
		type=str,
		default="speaker",
		help="Column for speaker name (default: speaker)",
	)
	p.add_argument(
		"--context",
		type=int,
		default=0,
		help=(
			"Number of surrounding speaker turns to include as context on each side. "
			"Adds data.prev_text and data.next_text. (default: 0)"
		),
	)
	return p.parse_args(argv)


def _safe_int(x: Any) -> Optional[int]:
	try:
		if x is None:
			return None
		return int(x)
	except Exception:
		return None


def _generate_stable_id(debate_id: str, row_number: int) -> int:
	"""Generate a stable numeric ID from date and row number.
	
	Debate IDs are formatted as YYYYMMDD_speaker1_speaker2.
	Extract the date and combine with the CSV row number for the ID.
	"""
	# Extract date from debate_id (first 8 chars: YYYYMMDD)
	date_str = debate_id[:8] if len(debate_id) >= 8 else "00000000"
	# Create ID: dateXXXXX where XXXXX is zero-padded row number
	# e.g., "20240910" with row 42 -> 202409100042
	return int(f"{date_str}{row_number:05d}")


def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)
	if not args.input_csv.exists():
		raise FileNotFoundError(f"Input not found: {args.input_csv}")

	out = args.out
	if out is None:
		out = args.input_csv.with_suffix(args.input_csv.suffix + ".labelstudio.json")

	df = pd.read_csv(args.input_csv)
	
	# Validate required columns
	for col in [args.text_col, args.debate_col, args.speaker_col]:
		if col not in df.columns:
			raise KeyError(f"Missing column {col!r}. Available: {list(df.columns)!r}")

	context_n = int(args.context)
	if context_n < 0:
		raise ValueError("--context must be >= 0")

	# Collect valid rows in original CSV order so we can compute neighbors.
	items: List[Dict[str, Any]] = []
	for idx, rec in enumerate(df.to_dict(orient="records")):
		text_val = rec.get(args.text_col)
		if text_val is None:
			continue
		text = str(text_val).strip()
		if not text:
			continue

		debate_id = str(rec.get(args.debate_col, "")).strip()
		speaker = str(rec.get(args.speaker_col, "")).strip()

		# Skip rows without debate_id or speaker
		if not debate_id or not speaker:
			continue

		items.append({
			"row": int(idx),
			"text": text,
			"debate_id": debate_id,
			"speaker": speaker,
			"rec": rec,
		})

	tasks: List[Dict[str, Any]] = []
	for i, item in enumerate(items):
		debate_id = item["debate_id"]
		speaker = item["speaker"]
		row_num = item["row"]
		
		# Generate stable ID from date in debate_id and row number
		stable_id = _generate_stable_id(debate_id, row_num)
		data: Dict[str, Any] = {
			"id": stable_id,
			"text": item["text"],
			"debate_id": item["debate_id"],
			"speaker": item["speaker"],
			"row": int(item["row"]),
			"source_csv": args.input_csv.name,
		}

		if context_n > 0:
			prev_slice = items[max(0, i - context_n) : i]
			next_slice = items[i + 1 : i + 1 + context_n]
			data["prev_text"] = "\n---\n".join(
				f"{x['speaker']}: {x['text']}" for x in prev_slice
			)
			data["next_text"] = "\n---\n".join(
				f"{x['speaker']}: {x['text']}" for x in next_slice
			)

		tasks.append({"data": data})

	out.parent.mkdir(parents=True, exist_ok=True)
	out.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	print(f"Wrote {out} ({len(tasks)} tasks)")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
