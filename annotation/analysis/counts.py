import argparse
from ast import literal_eval
from pathlib import Path
from typing import Optional, Sequence
import pandas as pd
from rich.console import Console
from spacy.lang.en import English

console = Console()
nlp = English()
nlp.add_pipe("sentencizer")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count the number of turns and words for each debate and speaker in a speaker-turn CSV."
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input CSV with debate_id, speaker, and text columns",
    )

    parser.add_argument(
        "--debate-col",
        type=str,
        default="debate_id",
        help="Debate id column name (default: debate_id)",
    )

    parser.add_argument(
        "--speaker-col",
        type=str,
        default="speaker",
        help="Speaker column name (default: speaker)",
    )

    parser.add_argument(
        "--text-col",
        type=str,
        default="text",
        help="Text column name (default: text)",
    )

    return parser.parse_args(argv)


def count_turns(
    df: pd.DataFrame,
    debate_col: str = "debate_id",
    speaker_col: str = "speaker",
    text_col: str = "text",
) -> pd.DataFrame:
    """Count the number of turns and words for each debate and speaker."""
    # Count turns
    debate_list = list(df[debate_col].unique())
    debate_list.sort()
    console.print(f"Total debates: {len(debate_list)}")
    console.print(f"Total turns: {len(df)}")

    # Per debate
    for debate_id in debate_list:
        debate_df = df[df[debate_col] == debate_id]
        speaker_counts = debate_df[speaker_col].value_counts()
        console.print(
            f"Debate {debate_id}: {len(debate_df)} turns ({len(debate_df) / len(df) * 100:.2f}%)"
        )
        for speaker, count in speaker_counts.items():
            console.print(
                f"  {speaker}: {count} turns ({count / len(debate_df) * 100:.2f}%)"
            )


def count_words_per_debate(df: pd.DataFrame, text_col: str = "text") -> None:
    """Count the number of words in the text column for each debate."""
    debate_list = list(df["debate_id"].unique())
    debate_list.sort()

    total_words = 0
    for debate_id in debate_list:
        debate_df = df[df["debate_id"] == debate_id]
        word_count = debate_df[text_col].apply(lambda x: len(str(x).split())).sum()
        console.print(f"Debate {debate_id}: {word_count} words")
        total_words += word_count

    console.print(f"Total words across all debates: {total_words}")


def count_total_sentences(df: pd.DataFrame, text_col: str = "text") -> None:
    """Count the total number of sentences in the text column."""
    total_sentences = df[text_col].apply(lambda x: len(list(nlp(str(x)).sents))).sum()
    console.print(f"Total sentences across all debates: {total_sentences}")


def count_spans(df: pd.DataFrame, spans_col: str = "spans") -> None:
    """Count the total number of spans in the spans column."""
    total_spans = (
        df[spans_col]
        .apply(lambda x: len(literal_eval(x)) if pd.notnull(x) else 0)
        .sum()
    )
    console.print(f"Total spans across all debates: {total_spans}")


def count_span_percentage(df: pd.DataFrame, spans_col: str = "spans") -> None:
    span_total_length = 0
    text_total_length = 0
    for i, row in df.iterrows():
        spans = literal_eval(row[spans_col]) if pd.notnull(row[spans_col]) else []
        text_length = len(row["text"]) if pd.notnull(row["text"]) else 0
        span_length = sum(span["end"] - span["start"] for span in spans)
        span_total_length += span_length
        text_total_length += text_length

    ratio = span_total_length / text_total_length if text_total_length > 0 else 0
    console.print(f"Span % of total text length: {ratio:.2%}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    # Load the CSV
    df = pd.read_csv(args.input_csv)

    # Validate columns
    required_columns = [args.debate_col, args.speaker_col, args.text_col]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Missing columns: {missing_columns!r}. Available columns: {list(df.columns)!r}"
        )

    # Count turns
    count_turns(
        df,
        debate_col=args.debate_col,
        speaker_col=args.speaker_col,
        text_col=args.text_col,
    )

    # Count total words
    count_words_per_debate(df, text_col=args.text_col)

    # Count total sentences
    count_total_sentences(df, text_col=args.text_col)

    # Count total spans
    count_spans(df, spans_col="spans")

    # Count span percentage
    count_span_percentage(df, spans_col="spans")


if __name__ == "__main__":
    main()
