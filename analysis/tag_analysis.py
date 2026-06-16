import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import ast

def _get_tag_counts(df):
    # Dictionary: debate_id -> Counter
    debate_counts = defaultdict(Counter)

    # Iterate rows
    for _, row in df.iterrows():
        bio_tags = ast.literal_eval(row.iloc[-1])  # safely parse list string

        # Count tags in this row and accumulate
        debate_counts[row["debate_id"]].update(bio_tags)

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
    
    return plot_data

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
    parser = argparse.ArgumentParser(description="Analyze BIO tag distribution in debates.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file containing debate data.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not Path(args.input_file).exists():
        raise FileNotFoundError(f"Input not found: {args.input_file}")
    
    outdir = Path(__file__).resolve().parent / "output"
    outdir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(args.input_file)
    plot_data = _get_tag_counts(df)
    plot_tag_distribution(plot_data, outdir)
if __name__ == "__main__":
    main()