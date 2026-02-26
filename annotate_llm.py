"""Span-annotate sentences for check-worthiness using an LLM.

Reads a CSV containing a column of sentences, prompts an LLM to identify
check-worthy spans (verifiable factual claims), and writes a new CSV containing
the original sentence plus span annotations.

Default runtime targets Ollama's local HTTP API with Llama 3.2 3B.

Example:
	python annotate_llm.py transcripts/20240910_trump_harris.csv --out annotated.csv

Prereqs (Ollama):
	1) Install Ollama: https://ollama.com
	2) Pull model: ollama pull llama3.2:3b
	3) Ensure Ollama is running (default: http://localhost:11434)
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


# Keep the prompt in one place so you can tweak it later.
# This is the “example query” the user asked for.
EXAMPLE_QUERY_TEMPLATE = (
	"You are a meticulous annotation assistant. "
	"Task: Identify check-worthy spans in the sentence below. "
	"A check-worthy span is a verifiable factual claim (numbers, dates, events, "
	"attributions, policies, measurable comparisons). "
	"Do NOT mark opinions, insults, vague rhetoric, or pure value judgments unless "
	"they contain a concrete factual claim.\n\n"
	"Return ONLY valid JSON with this exact schema:\n"
	"{\n"
	"  \"check_worthy\": true|false,\n"
	"  \"spans\": [\n"
	"    {\"start\": 0, \"end\": 0, \"text\": \"...\", \"reason\": \"...\"}\n"
	"  ]\n"
	"}\n\n"
	"Rules:\n"
	"- Indices are 0-based character offsets into the ORIGINAL sentence; end is exclusive.\n"
	"- Each span text MUST exactly match sentence[start:end].\n"
	"- Spans must not overlap; prefer the smallest span that captures the claim.\n"
	"- If no check-worthy spans exist: check_worthy=false and spans=[].\n\n"
	"Sentence:\n{sentence}"
)


@dataclass(frozen=True)
class Span:
	start: int
	end: int
	text: str
	reason: str


@dataclass(frozen=True)
class Annotation:
	check_worthy: bool
	spans: List[Span]

	def to_public_dict(self) -> Dict[str, Any]:
		return {
			"check_worthy": bool(self.check_worthy),
			"spans": [
				{
					"start": s.start,
					"end": s.end,
					"text": s.text,
					"reason": s.reason,
				}
				for s in self.spans
			],
		}


def _http_json(url: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
	body = json.dumps(payload).encode("utf-8")
	req = urllib.request.Request(
		url,
		data=body,
		headers={"Content-Type": "application/json"},
		method="POST",
	)
	with urllib.request.urlopen(req, timeout=timeout_s) as resp:
		data = resp.read().decode("utf-8", errors="replace")
	return json.loads(data)


def _extract_json_object(text: str) -> str:
	"""Extract a JSON object from model text.

	We instruct the model to return ONLY JSON, but in practice it sometimes wraps
	it in code fences or preambles. This tries to recover the first {...} block.
	"""
	text = text.strip()
	if not text:
		raise ValueError("Empty model response")

	# Remove common code fence wrappers.
	if text.startswith("```"):
		lines = text.splitlines()
		# Drop the opening ``` or ```json
		lines = lines[1:]
		# Drop closing ``` if present
		if lines and lines[-1].strip().startswith("```"):
			lines = lines[:-1]
		text = "\n".join(lines).strip()

	# Find the first '{' and the last '}' and try that slice.
	start = text.find("{")
	end = text.rfind("}")
	if start == -1 or end == -1 or end <= start:
		raise ValueError(f"No JSON object found in response: {text[:200]!r}")
	return text[start : end + 1]


def _spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
	return not (a[1] <= b[0] or b[1] <= a[0])


def _validate_and_normalize(sentence: str, obj: Dict[str, Any]) -> Annotation:
	check_worthy = bool(obj.get("check_worthy", False))
	spans_raw = obj.get("spans", [])
	if spans_raw is None:
		spans_raw = []
	if not isinstance(spans_raw, list):
		raise ValueError("'spans' must be a list")

	spans: List[Span] = []
	intervals: List[Tuple[int, int]] = []

	for item in spans_raw:
		if not isinstance(item, dict):
			continue

		start = item.get("start")
		end = item.get("end")
		text = item.get("text")
		reason = item.get("reason")

		if not isinstance(start, int) or not isinstance(end, int):
			continue
		if start < 0 or end < 0 or end < start:
			continue
		if end > len(sentence):
			continue

		if not isinstance(text, str):
			text = sentence[start:end]
		if not isinstance(reason, str):
			reason = ""

		# Enforce exact match.
		expected = sentence[start:end]
		if text != expected:
			# Try to repair by using the computed substring.
			text = expected

		interval = (start, end)
		if any(_spans_overlap(interval, prev) for prev in intervals):
			continue

		intervals.append(interval)
		spans.append(Span(start=start, end=end, text=text, reason=reason.strip()))

	# If spans exist, force check_worthy True.
	if spans:
		check_worthy = True

	# Sort spans by start.
	spans.sort(key=lambda s: (s.start, s.end))
	return Annotation(check_worthy=check_worthy, spans=spans)


class OllamaClient:
	def __init__(self, base_url: str, model: str, timeout_s: int = 60):
		self.base_url = base_url.rstrip("/")
		self.model = model
		self.timeout_s = timeout_s

	def annotate(self, sentence: str, prompt_template: str, temperature: float) -> Annotation:
		prompt = prompt_template.format(sentence=sentence)
		payload = {
			"model": self.model,
			"stream": False,
			"messages": [
				{
					"role": "system",
					"content": "You output only JSON. No extra text.",
				},
				{"role": "user", "content": prompt},
			],
			"options": {
				"temperature": float(temperature),
			},
		}

		resp = _http_json(f"{self.base_url}/api/chat", payload=payload, timeout_s=self.timeout_s)
		content = (resp.get("message") or {}).get("content")
		if not isinstance(content, str):
			raise ValueError("Unexpected Ollama response format: missing message.content")

		json_text = _extract_json_object(content)
		obj = json.loads(json_text)
		if not isinstance(obj, dict):
			raise ValueError("Expected a JSON object")
		return _validate_and_normalize(sentence, obj)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Span-annotate a sentence CSV for check-worthiness via Llama 3.2 3B.")
	parser.add_argument(
		"input",
		type=Path,
		help="Path to input .csv containing a column of sentences",
	)
	parser.add_argument(
		"--out",
		type=Path,
		default=None,
		help="Output .csv path (default: <input>_annotated.csv)",
	)
	parser.add_argument(
		"--sentence-col",
		type=str,
		default="sentence",
		help="Name of the sentence column in the input CSV (default: sentence)",
	)
	parser.add_argument(
		"--model",
		type=str,
		default=DEFAULT_MODEL,
		help=f"Ollama model name (default: {DEFAULT_MODEL})",
	)
	parser.add_argument(
		"--ollama-url",
		type=str,
		default=DEFAULT_OLLAMA_URL,
		help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=0.0,
		help="Sampling temperature (default: 0.0 for deterministic)",
	)
	parser.add_argument(
		"--sleep",
		type=float,
		default=0.0,
		help="Seconds to sleep between requests (default: 0.0)",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Only annotate the first N rows (useful for testing)",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Do not call the LLM; output empty spans for all rows",
	)
	return parser.parse_args(argv)


def _iter_sentences(df: pd.DataFrame, sentence_col: str, limit: Optional[int]) -> Iterable[Tuple[int, str]]:
	if sentence_col not in df.columns:
		raise KeyError(f"Sentence column not found: {sentence_col!r}. Available: {list(df.columns)!r}")

	count = 0
	for idx, value in df[sentence_col].items():
		if limit is not None and count >= limit:
			break
		if value is None:
			continue
		sentence = str(value).strip()
		if not sentence:
			continue
		yield int(idx), sentence
		count += 1


def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)

	if not args.input.exists():
		raise FileNotFoundError(f"Input file not found: {args.input}")

	out_path = args.out
	if out_path is None:
		out_path = args.input.with_name(f"{args.input.stem}_annotated.csv")

	df = pd.read_csv(args.input)

	client = OllamaClient(base_url=args.ollama_url, model=args.model)

	results: List[Dict[str, Any]] = []
	failed: int = 0

	for row_i, sentence in _iter_sentences(df, args.sentence_col, args.limit):
		if args.dry_run:
			ann = Annotation(check_worthy=False, spans=[])
		else:
			try:
				ann = client.annotate(sentence, prompt_template=EXAMPLE_QUERY_TEMPLATE, temperature=args.temperature)
			except (ValueError, json.JSONDecodeError, urllib.error.URLError, urllib.error.HTTPError) as e:
				failed += 1
				ann = Annotation(check_worthy=False, spans=[])
				results.append(
					{
						"row": row_i,
						"sentence": sentence,
						"check_worthy": ann.check_worthy,
						"spans": json.dumps(ann.to_public_dict()["spans"], ensure_ascii=False),
						"error": str(e),
					}
				)
				if args.sleep:
					time.sleep(args.sleep)
				continue

		results.append(
			{
				"row": row_i,
				"sentence": sentence,
				"check_worthy": ann.check_worthy,
				"spans": json.dumps(ann.to_public_dict()["spans"], ensure_ascii=False),
				"error": "",
			}
		)

		if args.sleep:
			time.sleep(args.sleep)

	out_df = pd.DataFrame(results)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_df.to_csv(out_path, index=False)

	# Helpful terminal summary.
	print(f"Wrote: {out_path} ({len(out_df)} rows). Failed rows: {failed}.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
