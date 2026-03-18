#!/usr/bin/env python3
"""
convert_jsonl_to_json.py

Convert a JSONL file to a pretty-printed JSON array for better readability.
Output is written to the same directory as the input, with .json extension.

Usage:
    python convert_jsonl_to_json.py <input.jsonl>
    python convert_jsonl_to_json.py file1.jsonl file2.jsonl ...
"""

import json
import os
import sys


def convert_one(jsonl_path):
    """Convert a single JSONL file to JSON array; write alongside input."""
    jsonl_path = os.path.abspath(jsonl_path)
    if not os.path.exists(jsonl_path):
        print(f"ERROR: File not found: {jsonl_path}", file=sys.stderr)
        return False

    if not jsonl_path.endswith(".jsonl"):
        base = jsonl_path
    else:
        base = jsonl_path[:-6]  # strip .jsonl

    out_path = base + ".json"
    dirname = os.path.dirname(jsonl_path)

    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    os.makedirs(dirname, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"  {len(records)} records → {out_path}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_jsonl_to_json.py <file.jsonl> [file2.jsonl ...]")
        sys.exit(1)

    ok = 0
    for path in sys.argv[1:]:
        if convert_one(path):
            ok += 1

    if ok < len(sys.argv) - 1:
        sys.exit(1)


if __name__ == "__main__":
    main()
