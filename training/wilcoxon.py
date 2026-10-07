"""Run a Wilcoxon signed-rank test for two paired arrays."""

import argparse

from scipy.stats import wilcoxon


def parse_args() -> argparse.Namespace:
    """Parse the paired arrays supplied on the command line."""
    parser = argparse.ArgumentParser(
        description="Perform a Wilcoxon signed-rank test on two paired arrays."
    )
    parser.add_argument(
        "--first",
        nargs="+",
        type=float,
        required=True,
        metavar="VALUE",
        help="Values in the first array.",
    )
    parser.add_argument(
        "--second",
        nargs="+",
        type=float,
        required=True,
        metavar="VALUE",
        help="Values in the second array.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.first) != len(args.second):
        raise SystemExit("The two arrays must have the same length.")

    result = wilcoxon(args.first, args.second)
    print(f"statistic: {result.statistic}")
    print(f"p-value: {result.pvalue}")


if __name__ == "__main__":
    main()
