#!/usr/bin/env python3
"""
Select qualitative-analysis cases from per-case metric CSV files.

Expected file pattern:
    <metrics_dir>/multiscale_eval_<model>_fold<FOLD_ID>_main.csv

This script:
1. Scans all models and folds in the metrics directory.
2. Loads per-case metrics from all CSV files.
3. Uses CATMIL as the target model.
4. Compares CATMIL against all other available models on the same fold/case.
5. Selects cases for qualitative visualization in three groups:
   - improvement: CATMIL clearly better than competitors
   - failure: CATMIL clearly worse than competitors
   - typical: CATMIL near its own median behavior
6. Prints selected case IDs.

The selection is driven mainly by:
- dice
- small_lesion_recall
- fn_lesion_count
- fp_volume_mm3
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


FILE_PATTERN = re.compile(
    r"^multiscale_eval_(?P<model>.+?)_fold(?P<fold>\d+)_main\.csv$"
)

REQUIRED_METRICS = [
    "dice",
    "hd95_mm",
    "small_lesion_recall",
    "lesion_f1",
    "fn_lesion_count",
    "fn_volume_fraction",
    "fp_volume_mm3",
]


@dataclass(frozen=True)
class CsvFileInfo:
    """Metadata extracted from a metric CSV filename."""

    model: str
    fold: int
    path: Path


@dataclass
class SelectionResult:
    """Selected cases for qualitative analysis."""

    improvement: pd.DataFrame
    failure: pd.DataFrame
    typical: pd.DataFrame


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Select cases for qualitative visualization."
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        required=True,
        help="Directory containing metric CSV files.",
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default="CATMIL",
        help="Model to analyze qualitatively. Default: CATMIL",
    )
    parser.add_argument(
        "--n_improve",
        type=int,
        default=4,
        help="Number of improvement cases to select.",
    )
    parser.add_argument(
        "--n_failure",
        type=int,
        default=3,
        help="Number of failure cases to select.",
    )
    parser.add_argument(
        "--n_typical",
        type=int,
        default=3,
        help="Number of typical cases to select.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="",
        help="Optional path to save selection summary CSV.",
    )
    return parser.parse_args()


def discover_csv_files(metrics_dir: Path) -> List[CsvFileInfo]:
    """Find all matching CSV files in the metrics directory."""
    csv_infos: List[CsvFileInfo] = []

    for path in sorted(metrics_dir.glob("multiscale_eval_*_fold*_main.csv")):
        match = FILE_PATTERN.match(path.name)
        if match is None:
            continue

        csv_infos.append(
            CsvFileInfo(
                model=match.group("model"),
                fold=int(match.group("fold")),
                path=path,
            )
        )

    if not csv_infos:
        raise FileNotFoundError(
            f"No matching CSV files found in: {metrics_dir}"
        )

    return csv_infos


def find_case_id_column(df: pd.DataFrame) -> str:
    """Infer the case identifier column."""
    candidates = [
        "case_id",
        "case",
        "subject_id",
        "subject",
        "image_id",
        "id",
        "name",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        "Could not find case ID column. Expected one of: "
        f"{candidates}. Found columns: {list(df.columns)}"
    )


def validate_metric_columns(df: pd.DataFrame, path: Path) -> None:
    """Ensure the required metrics exist."""
    missing = [col for col in REQUIRED_METRICS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}"
        )


def load_all_metrics(csv_infos: List[CsvFileInfo]) -> pd.DataFrame:
    """
    Load all CSVs into one long dataframe.

    Output columns include:
        fold, model, case_id, dice, hd95_mm,
        small_lesion_recall, lesion_f1,
        fn_lesion_count, fn_volume_fraction,
        fp_volume_mm3, ...
    """
    frames: List[pd.DataFrame] = []

    for info in csv_infos:
        df = pd.read_csv(info.path)
        case_id_col = find_case_id_column(df)
        validate_metric_columns(df, info.path)

        df = df.copy()
        df["case_id"] = df[case_id_col].astype(str)
        df["fold"] = info.fold
        df["model"] = info.model

        frames.append(df)

    all_df = pd.concat(frames, axis=0, ignore_index=True)

    return all_df


def build_case_comparison_table(
    all_df: pd.DataFrame,
    target_model: str,
) -> pd.DataFrame:
    """
    Build one-row-per-(fold, case_id) table for the target model.

    Each row contains:
    - target model metrics
    - average delta from target model to all other models
    - improvement / failure scores
    """
    target_df = all_df[all_df["model"] == target_model].copy()
    if target_df.empty:
        available = sorted(all_df["model"].unique().tolist())
        raise ValueError(
            f"Target model '{target_model}' not found. "
            f"Available models: {available}"
        )

    rows: List[Dict[str, object]] = []

    grouped_target = target_df.groupby(["fold", "case_id"], sort=True)

    for (fold, case_id), target_case in grouped_target:
        if len(target_case) != 1:
            raise ValueError(
                f"Expected exactly one row for target model case, "
                f"got {len(target_case)} for fold={fold}, case_id={case_id}"
            )

        target_row = target_case.iloc[0]
        competitors = all_df[
            (all_df["fold"] == fold)
            & (all_df["case_id"] == case_id)
            & (all_df["model"] != target_model)
        ].copy()

        if competitors.empty:
            continue

        delta_dice = (
            float(target_row["dice"]) - competitors["dice"].astype(float)
        ).mean()

        delta_small_recall = (
            float(target_row["small_lesion_recall"])
            - competitors["small_lesion_recall"].astype(float)
        ).mean()

        delta_fn_count = (
            competitors["fn_lesion_count"].astype(float)
            - float(target_row["fn_lesion_count"])
        ).mean()

        delta_fp_volume_mm3 = (
            competitors["fp_volume_mm3"].astype(float)
            - float(target_row["fp_volume_mm3"])
        ).mean()

        rows.append(
            {
                "fold": int(fold),
                "case_id": str(case_id),
                "n_competitors": int(len(competitors)),
                "target_dice": float(target_row["dice"]),
                "target_hd95_mm": float(target_row["hd95_mm"]),
                "target_small_lesion_recall": float(
                    target_row["small_lesion_recall"]
                ),
                "target_lesion_f1": float(target_row["lesion_f1"]),
                "target_fn_lesion_count": float(
                    target_row["fn_lesion_count"]
                ),
                "target_fn_volume_fraction": float(
                    target_row["fn_volume_fraction"]
                ),
                "target_fp_volume_mm3": float(
                    target_row["fp_volume_mm3"]
                ),
                "mean_delta_dice": float(delta_dice),
                "mean_delta_small_lesion_recall": float(
                    delta_small_recall
                ),
                "mean_delta_fn_lesion_count": float(delta_fn_count),
                "mean_delta_fp_volume_mm3": float(
                    delta_fp_volume_mm3
                ),
            }
        )

    case_df = pd.DataFrame(rows)
    if case_df.empty:
        raise ValueError(
            "No comparable cases found between target model and competitors."
        )

    return case_df


def zscore(series: pd.Series) -> pd.Series:
    """Safe z-score."""
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def score_cases(case_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score each case for:
    - improvement
    - failure
    - typicality
    """
    df = case_df.copy()

    # Improvement: target better than other models.
    # Positive is good for dice, small lesion recall, lower FN count.
    # FP increase is penalized.
    df["z_delta_dice"] = zscore(df["mean_delta_dice"])
    df["z_delta_small_recall"] = zscore(
        df["mean_delta_small_lesion_recall"]
    )
    df["z_delta_fn_count"] = zscore(df["mean_delta_fn_lesion_count"])
    df["z_delta_fp_volume_mm3"] = zscore(
        df["mean_delta_fp_volume_mm3"]
    )

    df["improvement_score"] = (
        0.35 * df["z_delta_small_recall"]
        + 0.30 * df["z_delta_fn_count"]
        + 0.25 * df["z_delta_dice"]
        + 0.10 * df["z_delta_fp_volume_mm3"]
    )

    df["failure_score"] = (
        0.35 * (-df["z_delta_small_recall"])
        + 0.30 * (-df["z_delta_fn_count"])
        + 0.25 * (-df["z_delta_dice"])
        + 0.10 * (-df["z_delta_fp_volume_mm3"])
    )

    median_dice = df["target_dice"].median()
    median_small_recall = df["target_small_lesion_recall"].median()
    median_fn = df["target_fn_lesion_count"].median()
    median_fp = df["target_fp_volume_mm3"].median()

    df["typical_distance"] = np.sqrt(
        (
            (df["target_dice"] - median_dice)
            / max(df["target_dice"].std(ddof=0), 1e-8)
        ) ** 2
        + (
            (
                df["target_small_lesion_recall"]
                - median_small_recall
            )
            / max(
                df["target_small_lesion_recall"].std(ddof=0),
                1e-8,
            )
        ) ** 2
        + (
            (df["target_fn_lesion_count"] - median_fn)
            / max(df["target_fn_lesion_count"].std(ddof=0), 1e-8)
        ) ** 2
        + (
            (df["target_fp_volume_mm3"] - median_fp)
            / max(df["target_fp_volume_mm3"].std(ddof=0), 1e-8)
        ) ** 2
    )

    return df


def select_top_cases(
    scored_df: pd.DataFrame,
    n_improve: int,
    n_failure: int,
    n_typical: int,
) -> SelectionResult:
    """Select improvement, failure, and typical cases."""
    improvement = (
        scored_df.sort_values(
            by=[
                "improvement_score",
                "target_small_lesion_recall",
                "target_dice",
            ],
            ascending=[False, False, False],
        )
        .head(n_improve)
        .copy()
    )

    used = set(zip(improvement["fold"], improvement["case_id"]))

    failure_pool = scored_df[
        ~scored_df.apply(
            lambda row: (row["fold"], row["case_id"]) in used,
            axis=1,
        )
    ]
    failure = (
        failure_pool.sort_values(
            by=["failure_score", "target_fp_volume_mm3"],
            ascending=[False, False],
        )
        .head(n_failure)
        .copy()
    )

    used.update(zip(failure["fold"], failure["case_id"]))

    typical_pool = scored_df[
        ~scored_df.apply(
            lambda row: (row["fold"], row["case_id"]) in used,
            axis=1,
        )
    ]
    typical = (
        typical_pool.sort_values(
            by=["typical_distance", "target_dice"],
            ascending=[True, False],
        )
        .head(n_typical)
        .copy()
    )

    return SelectionResult(
        improvement=improvement,
        failure=failure,
        typical=typical,
    )


def print_selected_cases(
    selection: SelectionResult,
    target_model: str,
) -> None:
    """Print selected case IDs and reasons."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    print()
    print("=" * 80)
    print(
        f"Selected qualitative-analysis cases for target model: "
        f"{target_model}"
    )
    print("=" * 80)

    print("\n[Improvement cases]")
    if selection.improvement.empty:
        print("  None")
    else:
        print(
            selection.improvement[
                [
                    "fold",
                    "case_id",
                    "improvement_score",
                    "target_dice",
                    "target_small_lesion_recall",
                    "target_fn_lesion_count",
                    "target_fp_volume_mm3",
                    "mean_delta_dice",
                    "mean_delta_small_lesion_recall",
                    "mean_delta_fn_lesion_count",
                    "mean_delta_fp_volume_mm3",
                ]
            ].to_string(index=False)
        )

    print("\n[Failure cases]")
    if selection.failure.empty:
        print("  None")
    else:
        print(
            selection.failure[
                [
                    "fold",
                    "case_id",
                    "failure_score",
                    "target_dice",
                    "target_small_lesion_recall",
                    "target_fn_lesion_count",
                    "target_fp_volume_mm3",
                    "mean_delta_dice",
                    "mean_delta_small_lesion_recall",
                    "mean_delta_fn_lesion_count",
                    "mean_delta_fp_volume_mm3",
                ]
            ].to_string(index=False)
        )

    print("\n[Typical cases]")
    if selection.typical.empty:
        print("  None")
    else:
        print(
            selection.typical[
                [
                    "fold",
                    "case_id",
                    "typical_distance",
                    "target_dice",
                    "target_small_lesion_recall",
                    "target_fn_lesion_count",
                    "target_fp_volume_mm3",
                ]
            ].to_string(index=False)
        )

    print("\n[Case IDs only]")
    all_selected = pd.concat(
        [
            selection.improvement.assign(category="improvement"),
            selection.failure.assign(category="failure"),
            selection.typical.assign(category="typical"),
        ],
        axis=0,
        ignore_index=True,
    )

    for _, row in all_selected.iterrows():
        print(
            f"{row['category']}: fold={int(row['fold'])}, "
            f"case_id={row['case_id']}"
        )


def save_selection_csv(
    selection: SelectionResult,
    out_csv: Path,
) -> None:
    """Save selected cases to CSV."""
    out_df = pd.concat(
        [
            selection.improvement.assign(category="improvement"),
            selection.failure.assign(category="failure"),
            selection.typical.assign(category="typical"),
        ],
        axis=0,
        ignore_index=True,
    )
    out_df.to_csv(out_csv, index=False)


def main() -> None:
    """Run case selection."""
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)

    csv_infos = discover_csv_files(metrics_dir)
    all_df = load_all_metrics(csv_infos)

    print(f"Found {len(csv_infos)} CSV files.")
    print(f"Models: {sorted(all_df['model'].unique().tolist())}")
    print(f"Folds: {sorted(all_df['fold'].unique().tolist())}")
    print(f"Target model: {args.target_model}")

    case_df = build_case_comparison_table(
        all_df=all_df,
        target_model=args.target_model,
    )

    scored_df = score_cases(case_df)

    selection = select_top_cases(
        scored_df=scored_df,
        n_improve=args.n_improve,
        n_failure=args.n_failure,
        n_typical=args.n_typical,
    )

    print_selected_cases(
        selection=selection,
        target_model=args.target_model,
    )

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        save_selection_csv(selection, out_csv)
        print(f"\nSaved selection CSV to: {out_csv}")


if __name__ == "__main__":
    main()