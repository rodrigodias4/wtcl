"""Convert a speaker-turn CSV into Label Studio JSON tasks.

This script handles the output from `parse_transcript.py`, which produces CSVs with:
  debate_id, speaker, text

Label Studio expects a list of tasks, where each task has a `data` dict.

We preserve the debate_id and speaker in task data so exports can be mapped back to the original debate.

Example:
  python labelstudio/ls_prepare_tasks.py transcripts/all_transcripts.csv \
    --out labelstudio/tasks_all_debates.json

To preserve existing annotations from a Label Studio export while regenerating
tasks with the current stable IDs:
	python labelstudio/ls_prepare_tasks.py transcripts/all_transcripts.csv \
		--merge-export labelstudio/project-1-at-YYYY-MM-DD.json \
		--out labelstudio/tasks_all_debates_merged.json

Then in Label Studio: Import -> JSON -> select the generated tasks file.
"""

from __future__ import annotations

import argparse
import copy
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
		"-m",
		"--merge-export",
		type=Path,
		default=None,
		help=(
			"Optional Label Studio export JSON to merge annotations from. "
			"Rows are matched by debate_id, speaker, and text."
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


def _generate_stable_id(debate_id: str, turn_index: int) -> int:
	"""Generate a stable numeric ID from date and debate turn index.
	
	Debate IDs are formatted as YYYYMMDD_speaker1_speaker2.
	Extract the date and combine it with the per-debate turn index for the ID.
	"""
	# Extract date from debate_id (first 8 chars: YYYYMMDD)
	date_str = debate_id[:8] if len(debate_id) >= 8 else "00000000"
	# Create ID: dateXXXXX where XXXXX is zero-padded turn index within the debate
	# e.g., "20240910" with turn 42 -> 202409100042
	return int(f"{date_str}{turn_index:05d}")


def _task_match_key(task: Dict[str, Any], text_col: str, debate_col: str, speaker_col: str) -> Optional[tuple[str, str, str]]:
	data = task.get("data")
	if not isinstance(data, dict):
		return None
	debate_id = str(data.get(debate_col, "")).strip()
	speaker = str(data.get(speaker_col, "")).strip()
	text = str(data.get(text_col, "")).strip()
	if not debate_id or not speaker or not text:
		return None
	return (debate_id, speaker, text)


def _load_export_tasks(export_path: Path) -> List[Dict[str, Any]]:
	if not export_path.exists():
		raise FileNotFoundError(f"Merge export not found: {export_path}")
	with export_path.open("r", encoding="utf-8") as handle:
		loaded = json.load(handle)
	if isinstance(loaded, list):
		return loaded
	if isinstance(loaded, dict) and isinstance(loaded.get("tasks"), list):
		return loaded["tasks"]
	raise ValueError(
		f"Unsupported merge export format in {export_path}. Expected a list of tasks or a dict with a 'tasks' list."
	)


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
	turn_counters: Dict[str, int] = {}
	for i, item in enumerate(items):
		debate_id = item["debate_id"]
		speaker = item["speaker"]
		turn_index = turn_counters.get(debate_id, 0)
		turn_counters[debate_id] = turn_index + 1
		
		# Generate stable ID from date in debate_id and turn index within the debate
		stable_id = _generate_stable_id(debate_id, turn_index)
		data: Dict[str, Any] = {
			"id": stable_id,
			"text": item["text"],
			"debate_id": item["debate_id"],
			"speaker": item["speaker"],
			"row": int(item["row"]),
			"source_csv": args.input_csv.name,
		}

		tasks.append({"data": data})

	if args.merge_export is not None:
		export_tasks = _load_export_tasks(args.merge_export)
		new_tasks_by_key: Dict[tuple[str, str, str], List[Dict[str, Any]]] = {}
		for task in tasks:
			key = _task_match_key(task, args.text_col, args.debate_col, args.speaker_col)
			if key is None:
				continue
			new_tasks_by_key.setdefault(key, []).append(task)

		matched_count = 0
		for export_task in export_tasks:
			key = _task_match_key(export_task, args.text_col, args.debate_col, args.speaker_col)
			if key is None:
				continue
			queued_tasks = new_tasks_by_key.get(key)
			if not queued_tasks:
				continue

			new_task = queued_tasks.pop(0)
			if export_task.get("annotations"):
				new_task["annotations"] = copy.deepcopy(export_task["annotations"])
			if export_task.get("drafts"):
				new_task["drafts"] = copy.deepcopy(export_task["drafts"])
			if export_task.get("predictions"):
				new_task["predictions"] = copy.deepcopy(export_task["predictions"])
			matched_count += 1

		print(f"Merged annotations from {args.merge_export} into {matched_count} matching tasks")

	out.parent.mkdir(parents=True, exist_ok=True)
	out.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	print(f"Wrote {out} ({len(tasks)} tasks)")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
