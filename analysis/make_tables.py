"""Regenerate the paper tables that come from results/per_fold/ and results/stats/.

Every value is computed from the per-case CSVs (never from the rounded JSONs): a fold value
is the mean over the test scans, a table value is the mean and sample standard deviation of
the five fold values, and paired differences are read from
results/stats/significance_case_level.csv (written by analysis/stats_significance.py).
Numbers are rounded half up at the printed precision, and best / second-best values are
ranked before rounding.

For each table, results/tables/<label>.tex holds the tabular environment (caption and
label stay in the paper) and results/tables/<label>.csv the unrounded values.

Usage: python analysis/make_tables.py [per_fold_dir] [stats_dir] [out_dir]
"""
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PER_FOLD = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "per_fold"
STATS = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "results" / "stats"
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "results" / "tables"

LOSSES = [("CATMIL", "CATMIL (ours)"), ("nnUNet", "DiceCE"), ("Tversky", "Tversky"),
          ("FocalTversky", "FocalTversky"), ("CCDiceCE", "CC-DiceCE")]
# metric column -> (row label, higher is better)
METRICS = {
    "small_lesion_recall": ("Small Lesion Recall", True),
    "fn_lesion_count": ("FN Count", False),
    "fn_volume_fraction": ("FN Volume Fraction", False),
    "dice": ("DSC", True),
    "hd95_mm": ("HD95 (mm)", False),
    "lesion_f1": ("Lesion F1", True),
    "fp_volume_mm3": ("FP Volume (mm$^3$)", False),
    "fp_blob_volume_frac": ("FP Blob", False),
}
MAIN_ROWS = ["small_lesion_recall", "fn_lesion_count", "fn_volume_fraction", None,
             "dice", "hd95_mm", "lesion_f1", "fp_volume_mm3"]  # None: \midrule
PAIRED_DEC = {"small_lesion_recall": 3, "fn_lesion_count": 2, "dice": 3, "hd95_mm": 2,
              "lesion_f1": 3, "fp_volume_mm3": 0}


def rnd(x, decimals):
    """Round half up (as printed by hand in the paper), return a Decimal."""
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def fmt_sig(x, sig):
    """Round to `sig` significant figures, keeping trailing zeros (22.0, 2.57)."""
    d = Decimal(repr(float(x)))
    if d == 0:
        return "0"
    return str(rnd(x, max(sig - d.adjusted() - 1, 0)))


def fmt_mean(metric, x):
    if metric == "fn_lesion_count":
        return fmt_sig(x, 3)
    if metric == "fp_volume_mm3":
        return str(rnd(x, 0))
    if metric == "hd95_mm":
        return str(rnd(x, 2))
    return str(rnd(x, 3))


def fmt_std(metric, x):
    """Two decimals; one significant figure when two decimals would print 0.00."""
    if metric == "fp_volume_mm3":
        return str(rnd(x, 0))
    s = rnd(x, 2)
    return fmt_sig(x, 1) if s == 0 and x > 0 else str(s)


def mark(cell, rank):
    return {0: f"\\textbf{{{cell}}}", 1: f"\\underline{{{cell}}}"}.get(rank, cell)


def ranks(values, higher, n_marked):
    """0 for the best value, 1 for the second best, None otherwise (ranked unrounded)."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=higher)
    return {i: (r if r < n_marked else None) for r, i in enumerate(order)}


def fold_means(folder, name, metric):
    files = sorted(folder.glob(f"multiscale_eval_{name}_fold[0-4]_*.csv"))
    if len(files) != 5:
        raise FileNotFoundError(f"{folder}: expected 5 folds of {name}, found {len(files)}")
    return pd.Series([pd.read_csv(f)[metric].mean() for f in files])


def arrow(higher):
    return "$\\uparrow$" if higher else "$\\downarrow$"


def write(label, lines, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{label}.tex").write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(OUT / f"{label}.csv", index=False)
    print("wrote", OUT / f"{label}.tex")


def mean_std_table(label, folder, suffix=""):
    """Losses x main metrics, mean +- sample std over folds, best bold, second underlined."""
    head = " & ".join(f"\\textbf{{{t}}}" for _, t in LOSSES)
    lines = ["\\begin{tabular}{lccccc}", "\\toprule", f"\\textbf{{Metric}} & {head} \\\\",
             "\\midrule"]
    rows = []
    for metric in MAIN_ROWS:
        if metric is None:
            lines.append("\\midrule")
            continue
        name, higher = METRICS[metric]
        stats = [fold_means(folder, m + suffix, metric) for m, _ in LOSSES]
        means = [s.mean() for s in stats]
        rk = ranks(means, higher, 2)
        cells = []
        for i, ((model, _), s) in enumerate(zip(LOSSES, stats)):
            cells.append(mark(f"{fmt_mean(metric, s.mean())} $\\pm$ {fmt_std(metric, s.std(ddof=1))}",
                              rk[i]))
            rows.append({"metric": metric, "model": model, "mean": s.mean(), "std": s.std(ddof=1)})
        lines.append(f"{name} {arrow(higher)}\n& " + "\n& ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(label, lines, rows)


def paired_cell(r, dec):
    def num(x):
        s = str(rnd(x, dec))
        return s.replace("-", "$-$") if s.startswith("-") else "+" + s
    cell = (f"{num(r.delta)} [{num(r.ci_low)}, {num(r.ci_high)}] "
            f"({int(r.patients_favouring_method)}/{int(r.n_patients)})")
    return f"\\textbf{{{cell}}}" if (r.ci_low > 0 or r.ci_high < 0) else cell


def paired_table(label, blocks):
    """blocks: [(title, [(row label, setting, metric), ...]), ...]"""
    st = pd.read_csv(STATS / "significance_case_level.csv")
    refs = [m for m, _ in LOSSES[1:]]
    head = " & ".join(f"\\textbf{{vs.\\ {t}}}" for _, t in LOSSES[1:])
    lines = ["\\begin{tabular}{lllll}", "\\toprule", f"\\textbf{{Metric}} & {head} \\\\"]
    rows = []
    for title, block in blocks:
        lines += ["\\midrule", f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        for row_label, setting, metric in block:
            cells = []
            for ref in refs:
                sel = st[(st.setting == setting) & (st.family == "main") & (st.method == "CATMIL")
                         & (st.reference == ref) & (st.metric == metric)]
                if len(sel) != 1:
                    raise KeyError((setting, ref, metric))
                r = sel.iloc[0]
                cells.append(paired_cell(r, PAIRED_DEC[metric]))
                rows.append({"block": title, "row": row_label, "reference": ref, "delta": r.delta,
                             "ci_low": r.ci_low, "ci_high": r.ci_high,
                             "patients_favouring_method": r.patients_favouring_method})
            lines.append(f"{row_label} {arrow(METRICS[metric][1])} & " + " & ".join(cells) + " \\\\")
    lines +=["\\bottomrule", "\\end{tabular}"]
    write(label, lines, rows)


def ablation_table(label, entries, metrics, n_marked):
    """entries: [(row name, folder, file name)], one value per metric = mean over folds."""
    cols = " & ".join(c for _, c in metrics)
    lines = [f"\\begin{{tabular}}{{l{'c' * len(metrics)}}}", "\\toprule", f"Model & {cols} \\\\",
             "\\midrule"]
    table = {}
    for row, folder, name in entries:
        files = sorted(folder.glob(f"multiscale_eval_{name}_fold*_all.csv"))
        if not files:
            raise FileNotFoundError(f"{folder}: no files for {name}")
        table[row] = {m: pd.Series([pd.read_csv(f)[m].mean() for f in files]).mean()
                      for m, _ in metrics}
    marks = {}
    for m, _ in metrics:
        rk = ranks([table[r][m] for r, _, _ in entries], METRICS[m][1], n_marked)
        for i, (r, _, _) in enumerate(entries):
            marks[r, m] = rk[i]
    for row, _, _ in entries:
        cells = [mark(str(rnd(table[row][m], 2 if m == "fp_volume_mm3" else 4)), marks[row, m])
                 for m, _ in metrics]
        lines.append(f"{row} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(label, lines, [{"model": r, **v} for r, v in table.items()])


def main():
    ms_main, ms_all = PER_FOLD / "MSLesSeg" / "main", PER_FOLD / "MSLesSeg" / "all"
    lj = PER_FOLD / "Ljubljana" / "all"

    mean_std_table("loss_comparison_small_lesion", ms_main)
    mean_std_table("loss_comparison_small_lesion_pp5", ms_main, "_pp5")
    mean_std_table("external_eval_3dmrms", lj)

    ms, pp5, ext = "MSLesSeg", "MSLesSeg (pp5)", "3D-MR-MS"
    full = ["small_lesion_recall", "fn_lesion_count", "dice", "hd95_mm", "lesion_f1",
            "fp_volume_mm3"]
    paired_table("paired_differences", [
        ("MSLesSeg", [(METRICS[m][0], ms, m) for m in ["small_lesion_recall", "fn_lesion_count",
                                                        "lesion_f1"]]
         + [("Lesion F1, PP5", pp5, "lesion_f1")]),
        ("3D-MR-MS", [(METRICS[m][0], ext, m) for m in ["small_lesion_recall", "fn_lesion_count",
                                                         "lesion_f1"]]),
    ])
    paired_table("paired_differences_full", [
        (title, [(METRICS[m][0], setting, m) for m in full])
        for title, setting in [("MSLesSeg", ms), ("MSLesSeg, PP5", pp5), ("3D-MR-MS", ext)]
    ])

    ablation_table(
        "ablation_cat_mil", [(m, ms_all, m) for m in ["CAT", "MIL", "CATMIL"]],
        [("dice", "DSC"), ("hd95_mm", "HD95 $\\downarrow$"), ("lesion_f1", "Lesion F1"),
         ("small_lesion_recall", "Small Lesion Recall"), ("fp_volume_mm3", "FP Vol $\\downarrow$"),
         ("fp_blob_volume_frac", "FP Blob $\\downarrow$")], n_marked=1)
    ablation_table(
        "ablation_lambda",
        [(f"CATMIL{n[3:5]}{n[8:10]}", PER_FOLD / "MSLesSeg" / "lambda_sweep", n)
         for n in ["CAT01MIL01", "CAT01MIL02", "CAT02MIL01", "CAT02MIL02"]],
        [("dice", "DSC"), ("hd95_mm", "HD95 $\\downarrow$"), ("lesion_f1", "Lesion F1"),
         ("small_lesion_recall", "Small Recall"), ("fp_volume_mm3", "FP Vol $\\downarrow$")],
        n_marked=2)


if __name__ == "__main__":
    main()
