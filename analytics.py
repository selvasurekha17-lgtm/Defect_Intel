import argparse
from pathlib import Path

import pandas as pd

from backend import batch_analytics


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "data" / "clustered_defects.csv"
SUMMARY_PATH = ROOT / "data" / "batch_summary.csv"


def main():
    parser = argparse.ArgumentParser(description="Compare defect report counts between batches.")
    parser.add_argument("--baseline", help="Explicit baseline batch ID")
    parser.add_argument("--compare", help="Explicit batch ID to compare against the baseline")
    parser.add_argument("--minimum-count", type=int, default=5)
    parser.add_argument("--increase-percent", type=float, default=50)
    args = parser.parse_args()

    if bool(args.baseline) != bool(args.compare):
        parser.error("--baseline and --compare must be supplied together.")
    if args.minimum_count < 1 or args.increase_percent < 0:
        parser.error("Thresholds must be non-negative and minimum-count must be at least 1.")

    df = pd.read_csv(INPUT_PATH)
    summary, alerts = batch_analytics(
        df,
        previous_batch=args.baseline,
        current_batch=args.compare,
        min_current_count=args.minimum_count,
        increase_threshold=args.increase_percent,
    )

    print("\nDEFECTINTEL — BATCH REPORT COUNTS")
    print(summary)
    print("\nEARLY WARNING REPORT")
    if not args.baseline:
        print(
            "No warnings calculated: batch IDs are not assumed to be chronological. "
            "Pass --baseline and --compare to evaluate an explicit pair."
        )
    elif alerts:
        for alert in alerts:
            change = (
                "no baseline reports"
                if alert["increase_percent"] is None
                else f"{alert['increase_percent']}% report-count increase"
            )
            print(
                f"{alert['severity']} WARNING: Group {alert['cluster']} in batch "
                f"{alert['batch']} has {alert['current_count']} reports "
                f"({change}; baseline {alert['previous_count']})."
            )
    else:
        print("No warnings triggered for the selected batch comparison.")

    summary.to_csv(SUMMARY_PATH)
    print(f"\nSaved batch summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
