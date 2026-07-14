import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import ast
from transformers import AutoTokenizer
from rich.console import Console

console = Console()

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")


def encode(
    text: str,
    spans: list[dict],
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
) -> dict:
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        padding="do_not_pad",
        max_length=max_length,
    )
    word_ids = enc.word_ids()

    labels = ["O"] * (max(word_ids) + 1)

    def token_overlaps_span(tok_start, tok_end, span_start, span_end):
        return tok_start < span_end and tok_end > span_start

    # Assign BIO labels to tokens based on the provided spans
    for span in spans:
        span_start = span["start"]
        span_end = span["end"]

        token_indices = []

        for i, (tok_start, tok_end) in enumerate(enc["offset_mapping"]):
            if token_overlaps_span(tok_start, tok_end, span_start, span_end):
                token_indices.append(i)

        if not token_indices:
            continue

        # BIO tagging
        labels[word_ids[token_indices[0]]] = "B"
        for idx in token_indices[1:]:
            if idx > 0 and word_ids[idx] > word_ids[idx - 1]:
                labels[word_ids[idx]] = "I"

    return labels


def _get_tag_counts(df):
    # Dictionary: debate_id -> Counter
    debate_counts = defaultdict(Counter)
    claims_per_turn = defaultdict(int)

    # Iterate rows
    for _, row in df.iterrows():
        spans: list[dict] = ast.literal_eval(row["spans"])
        bio_tags = encode(row["text"], spans, tokenizer)

        # Count tags in this row and accumulate
        debate_counts[row["debate_id"]].update(bio_tags)

        # Count claims per turn
        claims_per_turn[len(spans)] = claims_per_turn.get(len(spans), 0) + 1

    # Convert to DataFrame
    plot_data = pd.DataFrame.from_dict(debate_counts, orient="index").fillna(0)

    # Ensure all BIO columns exist
    for tag in ["B", "I", "O"]:
        if tag not in plot_data.columns:
            plot_data[tag] = 0

    # Keep consistent order
    plot_data = plot_data[["B", "I", "O"]]

    # Sort by debate date (optional)
    plot_data = plot_data.sort_index()

    return plot_data, claims_per_turn


def plot_tag_distribution(plot_data: pd.DataFrame, output_dir: Path):
    ax = plot_data.plot(
        kind="bar",
        stacked=True,
        figsize=(14, 7),
        edgecolor="white",
    )

    plt.xlabel("Debate")
    plt.ylabel("Token Count")
    plt.title("BIO Tag Distribution per Debate")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Tag")
    plt.tight_layout()
    plt.savefig(output_dir / "bio_tag_distribution.png", dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze BIO tag distribution in debates."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input CSV file containing debate data.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.input_file).exists():
        raise FileNotFoundError(f"Input not found: {args.input_file}")

    outdir = Path(__file__).resolve().parent / "output"
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_file)
    plot_data, claims_per_turn = _get_tag_counts(df)
    console.print(f"Tag counts per debate:\n{plot_data}")
    plot_tag_distribution(plot_data, outdir)

    claims_per_turn_sorted = {
        k: v for k, v in sorted(claims_per_turn.items(), key=lambda item: item[0])
    }
    console.print(f"Claims per turn:\n{json.dumps(claims_per_turn_sorted, indent=4)}")

    turns_with_no_claims = claims_per_turn.get(0, 0)
    console.print(
        f"Turns with no claims: {turns_with_no_claims} ({(turns_with_no_claims / len(df)):.1%})"
    )

    turn_with_at_least_one_claim = sum(v for k, v in claims_per_turn.items() if k > 0)
    console.print(
        f"Turns with at least one claim: {turn_with_at_least_one_claim} ({(turn_with_at_least_one_claim / len(df)):.1%})"
    )


if __name__ == "__main__":
    main()
