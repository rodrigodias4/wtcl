import argparse
from pathlib import Path
from typing import Optional, Sequence
import pandas as pd

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

def count_turns_and_words(df: pd.DataFrame, debate_col: str = "debate_id", speaker_col: str = "speaker", text_col: str = "text") -> pd.DataFrame:
    """Count the number of turns and words for each debate and speaker."""
    # Count turns
    debate_list = list(df[debate_col].unique())
    debate_list.sort()
    print(f"Total debates: {len(debate_list)}")
    print(f"Total turns: {len(df)}")
    
    for debate_id in debate_list:
        debate_df = df[df[debate_col] == debate_id]
        speaker_counts = debate_df[speaker_col].value_counts()
        print(f"Debate {debate_id}: {len(debate_df)} turns ({len(debate_df) / len(df) * 100:.2f}%)")
        for speaker, count in speaker_counts.items():
            print(f"  {speaker}: {count} turns ({count / len(debate_df) * 100:.2f}%)")
    
def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    
    # Load the CSV
    df = pd.read_csv(args.input_csv)
    
    # Validate columns
    required_columns = [args.debate_col, args.speaker_col, args.text_col]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns: {missing_columns!r}. Available columns: {list(df.columns)!r}")
    
    # Count turns and words
    count_turns_and_words(df, debate_col=args.debate_col, speaker_col=args.speaker_col, text_col=args.text_col)

if __name__ == "__main__":
    main()