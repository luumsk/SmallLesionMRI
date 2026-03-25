python select_qual_cases.py \
  --metrics_dir metrics/MSLesSeg/latest_checkpoints \
  --target_model CATMIL \
  --n_improve 4 \
  --n_failure 3 \
  --n_typical 3 \
  --out_csv metrics/MSLesSeg/latest_checkpoints/qualitative_case_selection.csv