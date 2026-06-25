#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path


def read_header(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return None


def concat_csv(input_files, output_path):
    """
    Concatenate CSV files while ensuring that the header is consistent
    and written only once.
    """
    if not input_files:
        print("No files to concatenate.", file=sys.stderr)
        return 0

    headers = []
    for p in input_files:
        hdr = read_header(p)
        if hdr is None:
            print(f"WARNING: {p.name} is empty; it will be skipped.", file=sys.stderr)
            continue
        headers.append((p, hdr))

    if not headers:
        print("No valid data to write.", file=sys.stderr)
        return 0

    ref_file, ref_header = headers[0]

    for p, hdr in headers[1:]:
        if hdr != ref_header:
            print(
                f"WARNING: The header of {p.name} does not match {ref_file.name}. "
                "The process will continue, but columns may be misaligned.",
                file=sys.stderr,
            )

    rows_written = 0

    with output_path.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(ref_header)

        for p in input_files:
            with p.open("r", newline="", encoding="utf-8") as in_f:
                reader = csv.reader(in_f)

                try:
                    next(reader)
                except StopIteration:
                    continue

                for row in reader:
                    writer.writerow(row)
                    rows_written += 1

    return rows_written


def get_fold_subdir(fold_type: str) -> str:
    """
    Return the folder name that contains the fold files.

    fold_type='cv':
        Ground-Truth/CV/fold_X.csv

    fold_type='non_cv':
        Ground-Truth/Non-CV/fold_X.csv
    """
    if fold_type == "cv":
        return "CV"

    if fold_type == "non_cv":
        return "Non-CV"

    raise ValueError(f"Unsupported fold type: {fold_type}")


def get_fold_filename(fold: int) -> str:
    """
    Return the fold CSV filename.
    """
    return f"fold_{fold}.csv"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate train.csv and test.csv from Soroll-IA ground-truth folds. "
            "Supports CV and Non-CV folds."
        )
    )

    parser.add_argument(
        "--dir", "-d",
        required=True,
        help=(
            "Base Ground-Truth directory containing CV/ and Non-CV/. "
            "Example: sorollia/Ground-Truth/"
        )
    )

    parser.add_argument(
        "--test-fold", "-t",
        type=int,
        required=True,
        choices=[0, 1, 2, 3, 4],
        help="Fold number to be used as the test set."
    )

    parser.add_argument(
        "--fold-type",
        choices=["cv", "non_cv"],
        required=True,
        help=(
            "Type of folds to use. "
            "'cv' uses Ground-Truth/CV/fold_X.csv. "
            "'non_cv' uses Ground-Truth/Non-CV/fold_X.csv."
        )
    )

    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory where train.csv and test.csv will be saved."
    )

    parser.add_argument(
        "--train-out",
        default="train.csv",
        help="Output CSV filename for the training set."
    )

    parser.add_argument(
        "--test-out",
        default="test.csv",
        help="Output CSV filename for the test set."
    )

    args = parser.parse_args()

    base = Path(args.dir)

    if not base.is_dir():
        print(f"ERROR: {base} is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    fold_subdir = get_fold_subdir(args.fold_type)
    folds_dir = base / fold_subdir

    if not folds_dir.is_dir():
        print(f"ERROR: {folds_dir} is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    folds = list(range(5))
    test_fold = args.test_fold

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = base / args.fold_type / f"fold_{test_fold}"

    out_dir.mkdir(parents=True, exist_ok=True)

    test_filename = get_fold_filename(test_fold)
    test_file = folds_dir / test_filename

    if not test_file.exists():
        print(f"ERROR: {test_file} was not found.", file=sys.stderr)
        sys.exit(1)

    train_files = []

    for f in folds:
        if f == test_fold:
            continue

        train_filename = get_fold_filename(f)
        train_file = folds_dir / train_filename

        if not train_file.exists():
            print(f"ERROR: {train_file} was not found.", file=sys.stderr)
            sys.exit(1)

        train_files.append(train_file)

    test_out = out_dir / args.test_out
    train_out = out_dir / args.train_out

    print("===================================")
    print(f"Fold type: {args.fold_type}")
    print(f"Folds directory: {folds_dir}")
    print(f"Test fold: {test_fold}")
    print(f"Test file: {test_file}")
    print("Training files:")
    for p in train_files:
        print(f"  - {p}")
    print(f"Output directory: {out_dir}")
    print("===================================")

    test_rows = concat_csv([test_file], test_out)
    print(f"[OK] Test: {test_out} ({test_rows} data rows)")

    train_rows = concat_csv(train_files, train_out)
    print(f"[OK] Train: {train_out} ({train_rows} data rows)")


if __name__ == "__main__":
    main()