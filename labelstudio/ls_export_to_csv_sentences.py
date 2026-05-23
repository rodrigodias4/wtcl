"""Convert a Label Studio JSON export into an annotated CSV (sentence-level).

This emits the same schema as `annotate_llm.py`:
  row,sentence,check_worthy,spans,error

Where `spans` is a JSON list of:
  {"start": int, "end": int, "text": str, "reason": str}

Assumptions:
- Tasks were imported using `ls_prepare_tasks_sentences.py`, which stores:
    data.text  (sentence)
    data.row   (stable row id)

Example:
  python labelstudio/ls_export_to_csv_sentences.py export.json --out transcripts/foo_annotated.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class Span:
	start: int
	end: int
	text: str
	reason: str


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Convert Label Studio export JSON to annotated CSV (sentences)")
	p.add_argument("export_json", type=Path, help="Label Studio JSON export file")
	p.add_argument(
		"--out",
		type=Path,
		default=None,
		help="Output CSV path (default: <export>.csv)",
	)
	p.add_argument(
		"--label",
		type=str,
		default="CheckWorthy",
		help="Label value to treat as check-worthy span (default: CheckWorthy)",
	)
	p.add_argument(
		"--take",
		type=str,
		default="latest",
		choices=["latest", "first"],
		help="Which annotation to use when multiple exist (default: latest)",
	)
	return p.parse_args(argv)


def _as_int(x: Any) -> Optional[int]:
	try:
		if x is None:
			return None
		return int(x)
	except Exception:
		return None


def _pick_annotation(task: Dict[str, Any], take: str) -> Optional[Dict[str, Any]]:
	anns = task.get("annotations")
	if anns is None:
		# Older Label Studio exports used `completions`.
		anns = task.get("completions")
	if not isinstance(anns, list) or not anns:
		return None
	if take == "first":
		return anns[0] if isinstance(anns[0], dict) else None
	# latest: choose the last dict-ish annotation
	for ann in reversed(anns):
		if isinstance(ann, dict):
			return ann
	return None


def _extract_reasons(result_list: List[Dict[str, Any]]) -> Dict[Tuple[int, int], str]:
	"""Map (start,end) -> reason string from per-region TextArea results."""
	out: Dict[Tuple[int, int], str] = {}
	for r in result_list:
		if not isinstance(r, dict):
			continue
		if r.get("type") != "textarea":
			continue
		v = r.get("value")
		if not isinstance(v, dict):
			continue
		start = _as_int(v.get("start"))
		end = _as_int(v.get("end"))
		if start is None or end is None:
			continue
		text_val = v.get("text")
		reason = ""
		if isinstance(text_val, list):
			reason = "\n".join(str(x) for x in text_val if str(x).strip())
		elif isinstance(text_val, str):
			reason = text_val
		reason = str(reason).strip()
		if reason:
			out[(start, end)] = reason
	return out


def _extract_spans(result_list: List[Dict[str, Any]], label_value: str) -> List[Span]:
	# Label Studio text NER spans come as `type: labels` with value:
	# {"start":..., "end":..., "text":..., "labels":[...]}
	reasons = _extract_reasons(result_list)
	spans: List[Span] = []
	seen: set[Tuple[int, int]] = set()

	for r in result_list:
		if not isinstance(r, dict):
			continue
		if r.get("type") != "labels":
			continue
		v = r.get("value")
		if not isinstance(v, dict):
			continue

		labels = v.get("labels")
		if isinstance(labels, list):
			labels_list = [str(x) for x in labels]
		else:
			labels_list = []

		if label_value not in labels_list:
			continue

		start = _as_int(v.get("start"))
		end = _as_int(v.get("end"))
		text = v.get("text")
		if start is None or end is None:
			continue
		if not isinstance(text, str):
			text = "" if text is None else str(text)
		key = (start, end)
		if key in seen:
			continue
		seen.add(key)

		reason = reasons.get(key, "")
		spans.append(Span(start=start, end=end, text=text, reason=reason))

	spans.sort(key=lambda s: (s.start, s.end))
	return spans


def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)
	if not args.export_json.exists():
		raise FileNotFoundError(f"Export not found: {args.export_json}")

	out = args.out
	if out is None:
		out = args.export_json.with_suffix(".csv")

	loaded = json.loads(args.export_json.read_text(encoding="utf-8"))
	# Label Studio commonly exports as a list of tasks, but some versions wrap it.
	if isinstance(loaded, dict):
		if isinstance(loaded.get("tasks"), list):
			tasks = loaded["tasks"]
		elif isinstance(loaded.get("data"), list):
			# Some integrations use {"data": [...tasks...]}
			tasks = loaded["data"]
		else:
			raise ValueError("Unrecognized export JSON dict shape; expected key 'tasks' or 'data'")
	elif isinstance(loaded, list):
		tasks = loaded
	else:
		raise ValueError("Expected export JSON to be a list of tasks or a dict wrapper")

	rows: List[Dict[str, Any]] = []
	for task in tasks:
		if not isinstance(task, dict):
			continue

		data = task.get("data")
		if not isinstance(data, dict):
			data = {}

		row_id = _as_int(data.get("row"))
		text = data.get("text")
		if text is None:
			text = data.get("sentence")
		sentence = "" if text is None else str(text)

		if row_id is None:
			# Fallback: Label Studio internal id (not stable across imports)
			row_id = _as_int(task.get("id"))
		if row_id is None:
			continue

		ann = _pick_annotation(task, take=str(args.take))
		result_list: List[Dict[str, Any]] = []
		if isinstance(ann, dict):
			res = ann.get("result")
			if isinstance(res, list):
				result_list = [r for r in res if isinstance(r, dict)]

		spans = _extract_spans(result_list, label_value=str(args.label))
		rows.append(
			{
				"row": int(row_id),
				"sentence": sentence,
				"check_worthy": bool(spans),
				"spans": json.dumps(
					[
						{
							"start": s.start,
							"end": s.end,
							"text": s.text,
							"reason": s.reason,
						}
						for s in spans
					],
					ensure_ascii=False,
				),
				"error": "",
			}
		)

	rows.sort(key=lambda r: r["row"])
	out_df = pd.DataFrame(rows, columns=["row", "sentence", "check_worthy", "spans", "error"])
	out.parent.mkdir(parents=True, exist_ok=True)
	out_df.to_csv(out, index=False)
	print(f"Wrote {out} ({len(out_df)} rows)")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
