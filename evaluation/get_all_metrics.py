import json
import argparse
from pathlib import Path

import pandas as pd


def load_json_files(json_dir: str) -> list[dict]:
    """Load all JSON files from a directory."""
    json_dir = Path(json_dir)
    json_files = sorted(json_dir.glob("*.json"))

    records = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            records.append(json.load(f))

    return records


def add_mean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add one row per model with:
      - same model_name
      - fold = 'mean'
      - each numeric metric = mean across that model's folds
    """
    if "model_name" not in df.columns or "fold" not in df.columns:
        raise ValueError("Input data must contain 'model_name' and 'fold'.")

    mean_rows = []

    for model_name, group in df.groupby("model_name", sort=False):
        # Keep only real fold rows, exclude any existing aggregate rows
        group_folds = group[group["fold"].astype(str).str.fullmatch(r"\d+")]

        if group_folds.empty:
            continue

        row = {
            "model_name": model_name,
            "fold": "mean",
        }

        for col in df.columns:
            if col in {"model_name", "fold"}:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                row[col] = group_folds[col].mean()
            else:
                row[col] = ""

        mean_rows.append(row)

    if not mean_rows:
        return df

    df_mean = pd.DataFrame(mean_rows)
    return pd.concat([df, df_mean], ignore_index=True)


def sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by model_name, then fold 0-4, then mean."""
    def fold_sort_key(value):
        value = str(value)
        if value.isdigit():
            return (0, int(value))
        if value == "mean":
            return (1, 999)
        return (2, 999)

    df = df.copy()
    df["_fold_sort"] = df["fold"].map(fold_sort_key)
    df = df.sort_values(
        by=["model_name", "_fold_sort"],
        kind="stable",
    ).drop(columns="_fold_sort")

    return df


def save_to_csv(records: list[dict], output_csv: str) -> None:
    """Save all JSON records and model mean rows into one CSV."""
    df = pd.DataFrame(records)

    df = add_mean_rows(df)
    df = sort_rows(df)

    priority_cols = ["model_name", "fold"]
    other_cols = [col for col in df.columns if col not in priority_cols]
    df = df[priority_cols + other_cols]

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved CSV to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge JSON files and add mean row per model."
    )
    parser.add_argument(
        "-i",
        "--json_dir",
        required=True,
        help="Directory containing JSON files.",
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        required=True,
        help="Path to output CSV file.",
    )
    args = parser.parse_args()

    records = load_json_files(args.json_dir)
    save_to_csv(records, args.output_csv)


if __name__ == "__main__":
    main()