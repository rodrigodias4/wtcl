"""Convert a sentence-level CSV into Label Studio JSON tasks.

This repo's transcripts are CSVs with a `sentence` column (one sentence per row).
Label Studio expects a list of tasks, where each task has a `data` dict.

We preserve a stable row id in `data.row` so exports can be mapped back to CSV.

Example:
  python labelstudio/ls_prepare_tasks_sentences.py transcripts/20240910_trump_harris.csv \
    --out labelstudio/tasks_20240910_trump_harris.json

Then in Label Studio: Import -> JSON -> select the generated tasks file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Prepare Label Studio tasks JSON from a sentence CSV")
	p.add_argument("input_csv", type=Path, help="Input CSV path")
	p.add_argument(
		"--out",
		type=Path,
		default=None,
		help="Output JSON path (default: <input>.labelstudio.json)",
	)
	p.add_argument(
		"--text-col",
		type=str,
		default="sentence",
		help="Column that contains the text to annotate (default: sentence)",
	)
	p.add_argument(
		"--row-col",
		type=str,
		default="row",
		help=(
			"Optional stable row id column. If absent, the CSV index is used. "
			"(default: row)"
		),
	)
	p.add_argument(
		"--include-cols",
		type=str,
		default="",
		help=(
			"Comma-separated extra columns to include in task data for display/filtering "
			"(default: none)"
		),
	)
	p.add_argument(
		"--context",
		type=int,
		default=0,
		help=(
			"Number of surrounding sentences to include as context on each side. "
			"Adds data.prev_text and data.next_text. (default: 0)"
		),
	)
	return p.parse_args(argv)


def _safe_int(x: Any) -> Optional[int]:
	try:
		if x is None:
			return None
		# pandas may give floats for int columns
		return int(x)
	except Exception:
		return None


def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)
	if not args.input_csv.exists():
		raise FileNotFoundError(f"Input not found: {args.input_csv}")

	out = args.out
	if out is None:
		out = args.input_csv.with_suffix(args.input_csv.suffix + ".labelstudio.json")

	df = pd.read_csv(args.input_csv)
	if args.text_col not in df.columns:
		raise KeyError(f"Missing text column {args.text_col!r}. Available: {list(df.columns)!r}")

	extra_cols: List[str] = [c.strip() for c in str(args.include_cols).split(",") if c.strip()]
	for c in extra_cols:
		if c not in df.columns:
			raise KeyError(f"Extra column not found: {c!r}. Available: {list(df.columns)!r}")

	row_col_present = args.row_col in df.columns
	context_n = int(args.context)
	if context_n < 0:
		raise ValueError("--context must be >= 0")

	# First, collect valid rows in original CSV order so we can compute neighbors.
	items: List[Dict[str, Any]] = []
	for idx, rec in enumerate(df.to_dict(orient="records")):
		text_val = rec.get(args.text_col)
		if text_val is None:
			continue
		text = str(text_val).strip()
		if not text:
			continue

		row_id: Optional[int]
		if row_col_present:
			row_id = _safe_int(rec.get(args.row_col))
		else:
			row_id = int(idx)
		if row_id is None:
			continue

		items.append({"row": int(row_id), "text": text, "rec": rec})

	tasks: List[Dict[str, Any]] = []
	for i, item in enumerate(items):
		# Skip rows with missing text
		if not item["text"]:
			continue
			
		data: Dict[str, Any] = {
			"text": item["text"],
			"row": int(item["row"]),
			"source_csv": args.input_csv.name,
		}

		if context_n > 0:
			prev_slice = items[max(0, i - context_n) : i]
			next_slice = items[i + 1 : i + 1 + context_n]
			data["prev_text"] = "\n".join(x["text"] for x in prev_slice)
			data["next_text"] = "\n".join(x["text"] for x in next_slice)

		rec = item["rec"]
		for c in extra_cols:
			val = rec.get(c)
			if val is None:
				continue
			data[c] = val

		tasks.append({"data": data})

	out.parent.mkdir(parents=True, exist_ok=True)
	out.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	print(f"Wrote {out} ({len(tasks)} tasks)")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
