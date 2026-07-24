from argparse import ArgumentParser
from ast import literal_eval
from pathlib import Path
import sys
from pandas import read_csv
from sklearn.metrics import cohen_kappa_score, f1_score
from rich.console import Console

sys.path.append((Path(__file__).resolve().parent.parent / "training").as_posix())
from train import encode, get_tokenizer
from utils import label_list
from plot_cm import compute_metrics_span_level

console = Console()


def parse_args():
    parser = ArgumentParser(description="Compute agreement metrics for annotations.")

    parser.add_argument("file_A", type=str, help="Path to the first annotation file.")

    parser.add_argument("file_B", type=str, help="Path to the second annotation file.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load annotations from the files
    df_A = read_csv(args.file_A)
    df_B = read_csv(args.file_B)

    # Ensure both dataframes have the same length
    if len(df_A) != len(df_B):
        raise ValueError("Annotation files must have the same number of entries.")

    labels_A = []
    labels_B = []
    tokenizer = get_tokenizer("FacebookAI/roberta-base")

    for idx in range(len(df_A)):
        row_A = df_A.iloc[idx]
        row_B = df_B.iloc[idx]

        spans_A = literal_eval(row_A["spans"])
        spans_B = literal_eval(row_B["spans"])

        # Encode the spans
        enc_A = encode(row_A["text"], spans_A, tokenizer)
        enc_B = encode(row_B["text"], spans_B, tokenizer)

        labels_A.append(
            [label for label, mask in zip(enc_A["labels"], enc_A["crf_mask"]) if mask]
        )
        labels_B.append(
            [label for label, mask in zip(enc_B["labels"], enc_B["crf_mask"]) if mask]
        )

    labels_A_flat = [label for sublist in labels_A for label in sublist]
    labels_B_flat = [label for sublist in labels_B for label in sublist]

    # Compute agreement metrics
    kappa = cohen_kappa_score(labels_A_flat, labels_B_flat)
    f1 = f1_score(
        labels_A_flat,
        labels_B_flat,
        labels=list(range(len(label_list))),
        average="macro",
    )
    f1_classes = f1_score(
        labels_A_flat, labels_B_flat, labels=list(range(len(label_list))), average=None
    )
    exact_span = compute_metrics_span_level(labels_A, labels_B)

    console.print(f"Cohen's Kappa: {kappa}")
    console.print(f"F1 Score: {f1}")
    for id in range(len(label_list)):
        console.print(f"F1 Score for class {label_list[id]}: {f1_classes[id]}")

    console.print(f"Exact Span F1: {exact_span['f1']}")


if __name__ == "__main__":
    main()
