"""Convert a Label Studio JSON export into an annotated speaker-turn CSV.

This emits a schema with:
    id, debate_id, speaker, text, spans, error

Where `spans` is a JSON list of:
    {"start": int, "end": int, "text": str, "reason_text": str,
     "reason_form": [str], "reason_frame": [str], "reason_domain": [str]}

The three reason axes are kept separate in the span dictionary.

Assumptions:
- Tasks were imported using `ls_prepare_tasks.py`, which stores:
        data.text      (speaker text)
        data.debate_id (debate identifier)
        data.speaker   (speaker name)
        data.row       (row index)

Example:
  python labelstudio/ls_export_to_csv.py export.json --out transcripts/annotated_turns.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from rich.console import Console

console = Console()

import pandas as pd


@dataclass
class Span:
    start: int
    end: int
    text: str
    reason_text: str = ""
    reason_form: List[str] = field(default_factory=list)
    reason_frame: List[str] = field(default_factory=list)
    reason_domain: List[str] = field(default_factory=list)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Label Studio export JSON to annotated speaker-turn CSV"
    )
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


def _extract_reasons(
    result_list: List[Dict[str, Any]],
    reason_from_names: Tuple[str, ...] = (
        "reason",
        "reason_form",
        "reason_frame",
        "reason_domain",
    ),
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Map (start,end) -> reason text and per-axis choices."""
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for r in result_list:
        if not isinstance(r, dict):
            continue
        rtype = r.get("type")
        if rtype not in {"textarea", "choices"}:
            continue
        if r.get("from_name") not in reason_from_names:
            continue
        v = r.get("value")
        if not isinstance(v, dict):
            continue
        start = _as_int(v.get("start"))
        end = _as_int(v.get("end"))
        if start is None or end is None:
            continue
        key = (start, end)
        if key not in out:
            out[key] = {
                "text": "",
                "reason_form": [],
                "reason_frame": [],
                "reason_domain": [],
            }

        if rtype == "textarea":
            text_val = v.get("text")
            text = ""
            if isinstance(text_val, list):
                text = "\n".join(str(x) for x in text_val if str(x).strip())
            elif isinstance(text_val, str):
                text = text_val
            out[key]["text"] = str(text).strip()

        elif rtype == "choices":
            from_name = str(r.get("from_name") or "")
            choices = v.get("choices")
            parsed_choices: List[str] = []
            if isinstance(choices, list):
                parsed_choices = [str(x) for x in choices if str(x).strip()]
            elif isinstance(choices, str):
                parsed_choices = [choices]

            if parsed_choices and from_name in {
                "reason_form",
                "reason_frame",
                "reason_domain",
            }:
                existing_choices = out[key][from_name]
                for choice in parsed_choices:
                    if choice not in existing_choices:
                        existing_choices.append(choice)
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

        reason_info = reasons.get(
            key,
            {
                "text": "",
                "reason_form": [],
                "reason_frame": [],
                "reason_domain": [],
            },
        )
        spans.append(
            Span(
                start=start,
                end=end,
                text=text,
                reason_text=reason_info.get("text", ""),
                reason_form=reason_info.get("reason_form", []),
                reason_frame=reason_info.get("reason_frame", []),
                reason_domain=reason_info.get("reason_domain", []),
            )
        )

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
            raise ValueError(
                "Unrecognized export JSON dict shape; expected key 'tasks' or 'data'"
            )
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
        text_val = "" if text is None else str(text)

        debate_id = str(data.get("debate_id", "")).strip()
        speaker = str(data.get("speaker", "")).strip()
        custom_id = str(data.get("id")).strip()

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
                "id": custom_id,
                "debate_id": debate_id,
                "speaker": speaker,
                "text": text_val,
                "spans": json.dumps(
                    [
                        {
                            "start": s.start,
                            "end": s.end,
                            "text": s.text,
                            "reason_text": s.reason_text,
                            "reason_form": s.reason_form,
                            "reason_frame": s.reason_frame,
                            "reason_domain": s.reason_domain,
                        }
                        for s in spans
                    ],
                    ensure_ascii=False,
                ),
            }
        )

    deduped_rows: List[Dict[str, Any]] = []
    seen_keys: set[Tuple[str, str, str, str]] = set()
    duplicates_removed = 0
    for row in rows:
        key = (
            row["debate_id"],
            row["speaker"],
            row["text"],
            str(row["id"]),
        )
        if key in seen_keys:
            duplicates_removed += 1
            continue
        seen_keys.add(key)
        deduped_rows.append(row)

    deduped_rows.sort(key=lambda r: (r["debate_id"], r["speaker"], r["text"], r["id"]))
    out_df = pd.DataFrame(
        [
            {
                "id": row["id"],
                "debate_id": row["debate_id"],
                "speaker": row["speaker"],
                "text": row["text"],
                "spans": row["spans"],
            }
            for row in deduped_rows
        ],
        columns=[
            "id",
            "debate_id",
            "speaker",
            "text",
            "spans",
        ],
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.sort_values("id", inplace=True)
    out_df.to_csv(out, index=False)
    console.print(f"Removed {duplicates_removed} duplicate rows")
    console.print(f"Wrote {out} ({len(out_df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
