#!/usr/bin/env python3
"""
Goes through all CSV files in a folder, computes the average of every
numeric column in each file, and writes all the results into one
summary CSV — one row per input file, with the file name included.

Usage:
    python average_csvs.py /path/to/folder [-o output.csv]

If no folder is given, the current directory is used.
"""

import argparse
import csv
import glob
import os


def average_csv_file(filepath):
    """Return a dict of {column_name: average} for all numeric columns."""
    sums = {}
    counts = {}

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col, value in row.items():
                try:
                    num = float(value)
                except (TypeError, ValueError):
                    continue  # skip non-numeric columns/cells
                sums[col] = sums.get(col, 0.0) + num
                counts[col] = counts.get(col, 0) + 1

    return {col: sums[col] / counts[col] for col in sums}


def main():
    parser = argparse.ArgumentParser(description="Average all columns in every CSV in a folder.")
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing the CSV files")
    parser.add_argument("-o", "--output", default="averages_summary.csv", help="Output CSV file name")
    args = parser.parse_args()

    csv_paths = sorted(glob.glob(os.path.join(args.folder, "*.csv")))
    # don't accidentally re-process a previous output file sitting in the same folder
    csv_paths = [p for p in csv_paths if os.path.basename(p) != args.output]

    if not csv_paths:
        print(f"No CSV files found in: {args.folder}")
        return

    all_results = []
    all_columns = []  # preserve first-seen column order

    for path in csv_paths:
        averages = average_csv_file(path)
        averages["file_name"] = os.path.basename(path)
        all_results.append(averages)

        for col in averages:
            if col not in all_columns:
                all_columns.append(col)

    # put file_name first
    all_columns.remove("file_name")
    fieldnames = ["file_name"] + all_columns

    output_path = os.path.join(args.folder, args.output)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"Processed {len(csv_paths)} file(s). Averages written to: {output_path}")


if __name__ == "__main__":
    main()
