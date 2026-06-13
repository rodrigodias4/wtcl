from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import json

MODEL = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Convert spans to BIO format")
    parser.add_argument("input_file", type=str, help="Path to input CSV file")
    parser.add_argument("-t", "--tokenizer", type=str, default=MODEL, help="Tokenizer model name")
    parser.add_argument("--keep_token", action="store_true", help="Whether to zip the tokens with the BIO labels in the output")
    return parser.parse_args()


def spans_to_bio(text, spans, keep_token):
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offsets = encoding["offset_mapping"]

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

    return list(zip(tokens, labels)) if keep_token else labels

def main():
    args = parse_args()
    output_file = f"{args.input_file.split('.')[0]}_bio_{MODEL}.csv"
    print(f"Processing file: {args.input_file}")
    print(f"Using model: {MODEL}")
    print(f"Output will be saved to: {output_file}")

    with open(args.input_file, "r") as f:
        data = pd.read_csv(f)
    
    for index, row in tqdm(data.iterrows()):
        text = row["text"]
        spans = json.loads(row["spans"])
        bio_labels = spans_to_bio(text, spans, args.keep_token)
        data.at[index, "bio_labels"] = json.dumps(bio_labels, ensure_ascii=False)
    
    data.drop(columns=["spans","error","check_worthy"], inplace=True)
    data.to_csv(output_file, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()