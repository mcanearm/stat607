import pstats
import argparse


if __name__ == "__main__":
    # ChatGPT generated simple helper script for this.
    parser = argparse.ArgumentParser(
        description="Print profiling statistics from a .prof file."
    )
    parser.add_argument("profile_file", type=str, help="Path to the .prof file")
    parser.add_argument(
        "--sort",
        type=str,
        default="time",
        help="Sorting key for stats (e.g., 'calls', 'cumulative', 'time')",
    )
    parser.add_argument(
        "--top", type=int, default=50, help="Number of top results to print"
    )

    args = parser.parse_args()

    stats = pstats.Stats(args.profile_file)
    stats.sort_stats(args.sort)
    stats.print_stats(args.top)
