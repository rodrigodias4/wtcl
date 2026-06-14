from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import json

MODEL = "xlm-roberta-base"


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Convert spans to BIO format")
    parser.add_argument("input_file", type=str, help="Path to input CSV file")
    parser.add_argument(
        "-m", "--model", type=str, default=MODEL, help="Tokenizer model name"
    )
    parser.add_argument(
        "--keep_token",
        action="store_true",
        help="Whether to zip the tokens with the BIO labels in the output",
    )
    return parser.parse_args()


def chunk_bio(id, text, spans, tokenizer, stride, max_length=512, keep_token=False):
    """
    Converts a single example into span-safe chunks.
    """

    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)

    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # ----------------------------
    # Build token-level BIO labels
    # -----------------------------
    labels = ["O"] * len(tokens)

    for span in spans:
        span_start = span["start"]
        span_end = span["end"]

        matching_tokens = []

        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start < span_end and tok_end > span_start:
                matching_tokens.append(i)

        if matching_tokens:
            labels[matching_tokens[0]] = "B"
            for idx in matching_tokens[1:]:
                labels[idx] = "I"

    if len(input_ids) <= max_length:
        return [{
            "input_ids": input_ids,
            "labels": labels
        }]

    # -----------------------------
    # Identify safe chunk boundaries
    # -----------------------------
    def is_inside_span(idx, span_ranges):
        return any(s <= idx < e for s, e in span_ranges)

    span_ranges = []
    for span in spans:
        start, end = span["start"], span["end"]
        token_range = []
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start < end and tok_end > start:
                token_range.append(i)
        if token_range:
            span_ranges.append((token_range[0], token_range[-1] + 1))

    def build_sentence_boundaries(tokens):
        sentence_end_punct = {".", "!", "?"}

        boundaries = []

        for i, tok in enumerate(tokens):
            # XLM-R may split punctuation into its own token or attach it
            if tok in sentence_end_punct:
                boundaries.append(i + 1)  # boundary AFTER punctuation

        return boundaries
    
    sentence_boundaries = build_sentence_boundaries(tokens)
    
    def snap_to_sentence(boundary, sentence_boundaries, min_boundary=0):
        """
        Moves boundary left to nearest valid sentence boundary.
        """

        valid = [b for b in sentence_boundaries if b <= boundary]

        if not valid:
            return boundary

        return max(valid[-1], min_boundary)

    # -----------------------------
    # Chunking loop (span-aware sliding window)
    # -----------------------------
    chunks = []
    start = 0
    n = len(input_ids)
    
    def make_span_safe(boundary, span_ranges, min_boundary=0):
        """
        Moves boundary left until it is not inside any span.
        """
        initial_boundary = boundary
        while boundary > min_boundary and any(s <= boundary < e for s, e in span_ranges):
            boundary -= 1
        return boundary

    while start < n:

        end = min(start + max_length, n)

        sentence_end = snap_to_sentence(end, sentence_boundaries)
        end = make_span_safe(sentence_end, span_ranges)

        # safety fallback
        if end <= start:
            end = min(start + max_length, n)

        chunks.append({
            "input_ids": input_ids[start:end],
            "labels": labels[start:end]
        })

        next_start = snap_to_sentence(start + stride, sentence_boundaries, min_boundary=start + stride / 2)
        next_start = make_span_safe(next_start, span_ranges, min_boundary=start + stride / 2)  # don't move back more than half the stride to avoid infinite loops

        # enforce forward progress
        if next_start <= start:
            break

        start = next_start

    assert(len(chunks) > 0), "No chunks were created. Check the input text and spans."
    assert(all(len(c["input_ids"]) == len(c["labels"]) for c in chunks)), "Mismatch between input_ids and labels lengths in chunks."
    tqdm.write(f"Row {id}: Tokenized into {len(tokens)} tokens, {len(spans)} spans, resulting in {len(chunks)} chunks.")
    return chunks


def main():
    args = parse_args()
    output_file = f"{args.input_file.split('.')[0]}_bio_{args.model}.csv"
    print(f"Processing file: {args.input_file}")
    print(f"Using model: {args.model}")
    print(f"Output will be saved to: {output_file}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, model_max_length=512)

    with open(args.input_file, "r") as f:
        data = pd.read_csv(f)

    all_rows = []
    for index, row in tqdm(
        data.iterrows(), total=len(data), desc="Converting spans to BIO"
    ):
        text = row["text"]
        chunks = chunk_bio(
            row["id"],
            text=text,
            spans=json.loads(row["spans"]),
            tokenizer=tokenizer,
            stride=384,
            max_length=512,
            keep_token=args.keep_token,
        )
        new_rows = []

        for chunk_id, chunk in enumerate(chunks):
            new_rows.append(
                {
                    "debate_id": row["debate_id"] if "debate_id" in row else index,
                    "id": row["id"],
                    "chunk_id": chunk_id,
                    "text": tokenizer.decode(
                        chunk["input_ids"], skip_special_tokens=True
                    ),
                    "labels": json.dumps(chunk["labels"], ensure_ascii=False),
                }
            )

        all_rows.extend(new_rows)

    print(f"Processed {len(data)} rows into {len(all_rows)} chunks.")

    pd.DataFrame(all_rows).sort_values(["id", "chunk_id"]).to_csv(
        output_file, index=False
    )


if __name__ == "__main__":
    main()
