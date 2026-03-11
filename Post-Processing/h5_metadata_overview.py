#!/usr/bin/env python3
"""
Create a CSV overview of metadata (and structure) stored in HDF5 files under a given directory.

Usage:
  python h5_metadata_overview.py [--root PATH] [--output FILE]
  python h5_metadata_overview.py                    # uses raw_data/low_threshold, writes to stdout or overview.csv

Metadata = root-level attributes (f.attrs) of each .h5 / .h5part file.
Also reports structural summary: number of top-level groups/datasets.
"""

import argparse
import csv
import os
import sys

import h5py


def _attr_to_str(value):
    """Convert an HDF5 attribute value to a string for CSV."""
    if hasattr(value, "shape") and getattr(value, "size", 1) > 1:
        return ",".join(str(v) for v in value.flatten().tolist())
    return str(value)


def collect_file_metadata(filepath, root_path):
    """
    Open an HDF5 file and return (relative_path, attrs_dict, n_top_level_keys, error_message).
    On error attrs_dict and n_top_level_keys are None, error_message is set.
    """
    try:
        rel = os.path.relpath(filepath, root_path) if os.path.isabs(root_path) or not os.path.isabs(filepath) else filepath
    except ValueError:
        rel = filepath
    try:
        with h5py.File(filepath, "r") as f:
            attrs = {}
            for key in f.attrs:
                val = f.attrs[key]
                try:
                    attrs[key] = _attr_to_str(val)
                except Exception as e:
                    attrs[key] = f"<error: {e}>"
            n_keys = len(f.keys())
        return (rel, attrs, n_keys, None)
    except Exception as e:
        return (rel, None, None, str(e))


def find_h5_files(root_path):
    """Yield full paths of .h5 and .h5part files under root_path."""
    for dirpath, _, filenames in os.walk(root_path):
        for f in filenames:
            if f.endswith(".h5") or f.endswith(".h5part") or f.endswith(".hdf5"):
                yield os.path.join(dirpath, f)


def main():
    parser = argparse.ArgumentParser(
        description="Create CSV overview of HDF5 metadata under a directory."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="raw_data/low_threshold",
        help="Root directory to search for .h5 / .h5part / .hdf5 files (default: raw_data/low_threshold)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output CSV path. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory for resolving paths (default: current working directory). Used to make --root absolute.",
    )
    args = parser.parse_args()

    base = args.base_dir or os.getcwd()
    root_path = os.path.normpath(os.path.join(base, args.root))
    if not os.path.isdir(root_path):
        print(f"Error: root is not a directory: {root_path}", file=sys.stderr)
        sys.exit(1)

    # Collect metadata from all files
    all_attr_names = set()
    rows = []
    for path in sorted(find_h5_files(root_path)):
        rel, attrs, n_keys, err = collect_file_metadata(path, root_path)
        if attrs is None:
            rows.append({"file": rel, "n_top_level_keys": "", "error": err or ""})
            continue
        all_attr_names.update(attrs.keys())
        row = {"file": rel, "n_top_level_keys": n_keys, "error": ""}
        row.update(attrs)
        rows.append(row)

    attr_columns = sorted(all_attr_names)
    fieldnames = ["file", "n_top_level_keys", "error"] + attr_columns

    # Ensure every row has all attr columns (fill missing with "")
    for row in rows:
        for col in attr_columns:
            if col not in row:
                row[col] = ""

    out = open(args.output, "w", newline="", encoding="utf-8") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output and out is not sys.stdout:
            out.close()

    if args.output:
        print(f"Wrote {len(rows)} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
