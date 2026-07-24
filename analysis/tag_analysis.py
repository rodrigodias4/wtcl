import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import ast
from transformers import AutoTokenizer
from rich.console import Console

sys.path.append(str(Path(__file__).resolve().parent.parent / "training"))
from train import B_ID, encode
from utils import id2label

console = Console()

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")


def _get_tag_counts(df):
    # Dictionary: debate_id -> Counter
    debate_counts = defaultdict(Counter)
    claims_per_turn = defaultdict(int)

    # Iterate rows
    for _, row in df.iterrows():
        spans: list[dict] = ast.literal_eval(row["spans"])
        enc = encode(row["text"], spans, tokenizer)
        labels = [
            id2label[label]
            for label, mask in zip(enc["labels"], enc["crf_mask"])
            if mask == True
        ]

        # Count tags in this row and accumulate
        debate_counts[row["debate_id"]].update(labels)

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
        kind="bar", stacked=True, figsize=(14, 7), edgecolor="white", zorder=3
    )

    plt.xlabel("Debate")
    plt.ylabel("Token Count")
    plt.title("BIO Tag Distribution per Debate")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.5, zorder=0)
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
