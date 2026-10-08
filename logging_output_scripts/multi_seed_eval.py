import argparse
import csv
import statistics as st
import sys


def load_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return rows, reader.fieldnames


def numeric_columns(rows, fieldnames, exclude):
    """Return the list of columns that are fully numeric and not excluded."""
    numeric_cols = []
    for col in fieldnames:
        if col in exclude:
            continue
        try:
            for r in rows:
                float(r[col])
            numeric_cols.append(col)
        except (ValueError, TypeError):
            # column has non-numeric values (e.g. a label) -> skip
            continue
    return numeric_cols


def summarize(rows, cols):
    summary = {}
    for c in cols:
        vals = [float(r[c]) for r in rows]
        summary[c] = {
            "mean": st.mean(vals),
            "std": st.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
            "count": len(vals),
        }
    return summary


def print_summary(summary, n_rows):
    print(f"\nAveraged over {n_rows} runs\n")
    header = f"{'metric':<20}{'mean':>14}{'std':>14}{'min':>14}{'max':>14}"
    print(header)
    print("-" * len(header))
    for col, s in summary.items():
        print(f"{col:<20}{s['mean']:>14.6f}{s['std']:>14.6f}" f"{s['min']:>14.6f}{s['max']:>14.6f}")


def write_summary_csv(summary, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "min", "max", "count"])
        for col, s in summary.items():
            writer.writerow([col, s["mean"], s["std"], s["min"], s["max"], s["count"]])
    print(f"\nSummary written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Average numeric columns across experiment runs.")
    parser.add_argument("csv_path", help="Path to the input CSV file")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["seed"],
        help="Column names to exclude from averaging (default: seed)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the summary as a CSV file",
    )
    args = parser.parse_args()

    try:
        rows, fieldnames = load_rows(args.csv_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    cols = numeric_columns(rows, fieldnames, set(args.exclude))
    if not cols:
        print("No numeric columns found to average.", file=sys.stderr)
        sys.exit(1)

    summary = summarize(rows, cols)
    print_summary(summary, len(rows))

    if args.out:
        write_summary_csv(summary, args.out)


if __name__ == "__main__":
    main()
