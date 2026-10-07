import argparse
import json
from pathlib import Path

from numpy import mean, std
import pandas as pd

from plot_cm import compute_metrics_span_level
from utils import console, label2id
from train import compute_metrics_token_level, encode, get_tokenizer

DEFAULT_MODEL = "FacebookAI/roberta-base"


def _parse_json(value, field_name):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON list")
    return value


def _sentence_offsets(text, sentence_texts, debate_id, row_id):
    offsets = []
    search_start = 0
    for sentence_num, sentence_text in enumerate(sentence_texts):
        sentence_text = str(sentence_text)
        start = text.find(sentence_text, search_start)
        if start < 0:
            raise ValueError(
                f"Could not match sentence {sentence_num} for turn {row_id!r} "
                f"in debate {debate_id!r}"
            )
        end = start + len(sentence_text)
        offsets.append((start, end))
        search_start = end

    if text[search_start:].strip():
        raise ValueError(
            f"Sentence text does not cover the complete turn {row_id!r} "
            f"in debate {debate_id!r}: unmatched suffix starts at {search_start}"
        )
    return offsets


def _project_turn(original_row, sentence_rows, sentence_predictions, tokenizer):
    text = str(original_row["text"])
    sentence_texts = [row["text"] for row in sentence_rows]
    sentence_offsets = _sentence_offsets(
        text, sentence_texts, original_row["debate_id"], original_row["id"]
    )

    spans = _parse_json(original_row["spans"], "spans")
    full_token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    encoding = encode(text, spans, tokenizer, max_length=max(512, full_token_count))
    first_subword_indices = [
        index for index, keep in enumerate(encoding["crf_mask"]) if keep
    ]
    token_offsets = encoding["offset_mapping"]
    token_labels = [encoding["labels"][index] for index in first_subword_indices]

    token_predictions = []
    sentence_token_counts = [0] * len(sentence_offsets)
    sentence_index = 0
    for index in first_subword_indices:
        token_start, token_end = token_offsets[index]
        while (
            sentence_index < len(sentence_offsets)
            and token_start >= sentence_offsets[sentence_index][1]
        ):
            sentence_index += 1
        if sentence_index == len(sentence_offsets):
            raise ValueError(
                f"Token offset {token_offsets[index]} is outside the sentences "
                f"for turn {original_row['id']!r}"
            )
        sentence_start, sentence_end = sentence_offsets[sentence_index]
        if token_start < sentence_start or token_end > sentence_end:
            raise ValueError(
                f"Token offset {token_offsets[index]} crosses a sentence boundary "
                f"for turn {original_row['id']!r}"
            )
        sentence_prediction = int(sentence_predictions[sentence_index])
        if sentence_prediction not in (0, 1):
            raise ValueError(
                f"Sentence prediction must be binary for turn {original_row['id']!r}, "
                f"got {sentence_prediction}"
            )
        if sentence_prediction == 0:
            token_predictions.append(label2id["O"])
        elif sentence_token_counts[sentence_index] == 0:
            token_predictions.append(label2id["B"])
        else:
            token_predictions.append(label2id["I"])
        sentence_token_counts[sentence_index] += 1

    if sentence_index != len(sentence_offsets) - 1:
        raise ValueError(
            f"Not all sentences received tokens for turn {original_row['id']!r}"
        )
    if len(token_predictions) != len(token_labels):
        raise AssertionError(
            f"Projected predictions and BIO labels differ for turn {original_row['id']!r}: "
            f"{len(token_predictions)} != {len(token_labels)}"
        )
    return token_predictions, token_labels


def _sentence_bio(sentence_row, sentence_prediction, tokenizer):
    text = str(sentence_row["text"])
    spans = _parse_json(sentence_row["spans"], "sentence spans")
    full_token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    encoding = encode(text, spans, tokenizer, max_length=max(512, full_token_count))
    first_subword_indices = [
        index for index, keep in enumerate(encoding["crf_mask"]) if keep
    ]
    sentence_prediction = int(sentence_prediction)
    if sentence_prediction not in (0, 1):
        raise ValueError(
            f"Sentence prediction must be binary for sentence {sentence_row['id']!r}, "
            f"got {sentence_prediction}"
        )

    predicted_labels = []
    for token_index, _ in enumerate(first_subword_indices):
        if sentence_prediction == 0:
            predicted_labels.append(label2id["O"])
        elif token_index == 0:
            predicted_labels.append(label2id["B"])
        else:
            predicted_labels.append(label2id["I"])
    gold_labels = [encoding["labels"][index] for index in first_subword_indices]
    if len(predicted_labels) != len(gold_labels):
        raise AssertionError(
            f"Sentence BIO prediction and label lengths differ for row {sentence_row['id']!r}"
        )
    return predicted_labels, gold_labels


def project(
    sentence_preds_labels_path,
    dataset_sentences_path,
    dataset_spans_path,
    model_name,
    output_path,
):
    with Path(sentence_preds_labels_path).open() as file:
        sentence_preds_labels = json.load(file)
    sentence_predictions = sentence_preds_labels["preds"]
    sentence_labels = sentence_preds_labels.get("labels")
    if sentence_labels is None:
        raise ValueError("Sentence predictions file must contain a 'labels' object")

    sentence_df = pd.read_csv(dataset_sentences_path)
    spans_df = pd.read_csv(dataset_spans_path)
    required_sentence_columns = {"id", "debate_id", "sentence_num", "text", "label"}
    required_span_columns = {"id", "debate_id", "text", "spans"}
    if not required_sentence_columns <= set(sentence_df.columns):
        raise ValueError(
            f"Sentence dataset is missing columns: {required_sentence_columns - set(sentence_df.columns)}"
        )
    if not required_span_columns <= set(spans_df.columns):
        raise ValueError(
            f"Span dataset is missing columns: {required_span_columns - set(spans_df.columns)}"
        )

    tokenizer = get_tokenizer(model_name)
    output_preds = {}
    output_labels = {}
    sentence_output_preds = {}
    sentence_output_labels = {}

    for debate_id, original_debate in spans_df.groupby("debate_id", sort=False):
        debate_key = str(debate_id)
        if debate_key not in sentence_predictions or debate_key not in sentence_labels:
            raise ValueError(f"Missing predictions or labels for debate {debate_key!r}")
        debate_sentences = sentence_df[sentence_df["debate_id"] == debate_id]
        predictions = sentence_predictions[debate_key]
        labels = sentence_labels[debate_key]
        if len(predictions) != len(debate_sentences) or len(labels) != len(
            debate_sentences
        ):
            raise ValueError(
                f"Debate {debate_key!r} has {len(debate_sentences)} sentence rows, "
                f"{len(predictions)} predictions, and {len(labels)} labels"
            )

        sentence_prediction_by_key = {}
        sentence_label_by_key = {}
        for sentence_position, (_, sentence_row) in enumerate(
            debate_sentences.iterrows()
        ):
            key = (str(sentence_row["id"]), int(sentence_row["sentence_num"]))
            if key in sentence_prediction_by_key:
                raise ValueError(
                    f"Duplicate sentence key {key!r} in debate {debate_key!r}"
                )
            sentence_prediction_by_key[key] = predictions[sentence_position]
            sentence_label_by_key[key] = labels[sentence_position]

        output_preds[debate_key] = []
        output_labels[debate_key] = []
        sentence_output_preds[debate_key] = []
        sentence_output_labels[debate_key] = []
        for _, sentence_row in debate_sentences.iterrows():
            sentence_key = (
                str(sentence_row["id"]),
                int(sentence_row["sentence_num"]),
            )
            sentence_preds, sentence_gold = _sentence_bio(
                sentence_row,
                sentence_prediction_by_key[sentence_key],
                tokenizer,
            )
            sentence_output_preds[debate_key].append(sentence_preds)
            sentence_output_labels[debate_key].append(sentence_gold)
        for _, original_row in original_debate.iterrows():
            sentence_rows = sentence_df[
                (sentence_df["debate_id"] == debate_id)
                & (sentence_df["id"] == original_row["id"])
            ].sort_values("sentence_num")
            if sentence_rows.empty:
                raise ValueError(
                    f"No sentence rows found for turn {original_row['id']!r}"
                )
            sentence_keys = [
                (str(row["id"]), int(row["sentence_num"]))
                for row in sentence_rows.to_dict("records")
            ]
            if any(key not in sentence_prediction_by_key for key in sentence_keys):
                raise ValueError(
                    f"Sentence prediction alignment failed for turn {original_row['id']!r}"
                )
            sentence_predictions_for_turn = [
                sentence_prediction_by_key[key] for key in sentence_keys
            ]
            sentence_labels_for_turn = [
                sentence_label_by_key[key] for key in sentence_keys
            ]
            if list(sentence_rows["label"].astype(int)) != [
                int(value) for value in sentence_labels_for_turn
            ]:
                raise ValueError(
                    f"Sentence labels do not match the sentence dataset for turn {original_row['id']!r}"
                )
            projected_preds, projected_labels = _project_turn(
                original_row,
                sentence_rows.to_dict("records"),
                sentence_predictions_for_turn,
                tokenizer,
            )
            output_preds[debate_key].append(projected_preds)
            output_labels[debate_key].append(projected_labels)

    sentence_output_path = Path(output_path).with_name(
        "test_preds_labels_bio_sentences.json"
    )
    with sentence_output_path.open("w") as file:
        json.dump(
            {"preds": sentence_output_preds, "labels": sentence_output_labels},
            file,
            indent=4,
        )
    with Path(output_path).open("w") as file:
        json.dump({"preds": output_preds, "labels": output_labels}, file, indent=4)

    return output_preds, output_labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sentence_preds_labels",
        type=str,
        help="Path to sentence model predictions and labels.",
    )
    parser.add_argument(
        "--dataset-sentences",
        required=True,
        type=str,
        help="Path to the sentencized dataset CSV.",
    )
    parser.add_argument(
        "--dataset-spans",
        required=True,
        type=str,
        help="Path to the original span dataset CSV.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Tokenizer model used for the original BIO dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: test_preds_labels_bio.json beside the input).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path = args.output or str(
        Path(args.sentence_preds_labels).with_name("test_preds_labels_bio.json")
    )
    preds, labels = project(
        args.sentence_preds_labels,
        args.dataset_sentences,
        args.dataset_spans,
        args.model_name,
        output_path,
    )

    metrics = {debate: {} for debate in preds.keys()}

    for debate in preds.keys():
        metrics[debate]["test_metrics"] = compute_metrics_token_level(
            preds[debate], labels[debate]
        )
        metrics[debate]["test_metrics"]["span"] = compute_metrics_span_level(
            preds[debate], labels[debate]
        )

    metrics["overall"] = {"test": {}}
    for label in ["macro", "span", "B", "I", "O"]:
        # Initialize overall metrics dictionaries for each label
        metrics["overall"]["test"][label] = {"mean": {}, "std": {}}

        for metric in ["f1", "precision", "recall"]:
            test_values = [
                metrics[debate]["test_metrics"][label][metric]
                for debate in preds.keys()
            ]
            metrics["overall"]["test"][label]["mean"][metric] = mean(test_values)
            metrics["overall"]["test"][label]["std"][metric] = std(test_values)
    # console.print(f"Token-level metrics: {json.dumps(metrics, indent=4)}")

    with Path(output_path).with_name("results_token.json").open("w") as f:
        json.dump(metrics, f, indent=4)
