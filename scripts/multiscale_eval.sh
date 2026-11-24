#!/usr/bin/env bash
# Regenerate the per-fold, per-case metrics in results/per_fold/ from the saved predictions.
#
#   bash scripts/multiscale_eval.sh                      # every set
#   bash scripts/multiscale_eval.sh mslesseg_main lambda # only some sets
#
# Sets: mslesseg_main, mslesseg_all, lambda, ljubljana (see results/per_fold/README.md).
# Output goes to $OUT_ROOT (default results/per_fold_rerun, so the committed files are not
# overwritten). Set OUT_ROOT=results/per_fold to replace them.
# Post-processed masks (<model>_ppN: predicted components < N voxels removed) are always
# rebuilt from the raw predictions with evaluation/postprocess_min_size.py into $PP_ROOT
# (default $OUT_ROOT/pp_masks); the stored 3D-MR-MS *_pp5 folders are unfiltered copies.
# Missing prediction folders are skipped with a warning.
#
# Run from anywhere; paths are relative to the repository root.
set -euo pipefail
cd "$(dirname "$0")/.." && source scripts/config.sh

OUT_ROOT="${OUT_ROOT:-results/per_fold_rerun}"
PP_ROOT="${PP_ROOT:-$OUT_ROOT/pp_masks}"
PYTHON="${PYTHON:-python}"
MS_GT="${nnUNet_raw}/Dataset333_MSLesSeg/labelsTs"
MS_PRED="${PRED_ROOT}/final_checkpoints"
LJ_GT="${nnUNet_raw}/Dataset666_LjubljanaMS/labelsTs"
LJ_PRED="${LJ_PRED_ROOT:-$SLS_DATA_ROOT/SmallLesionMRI/LjubljanaMS}/final_checkpoints"
FOLDS=(0 1 2 3 4)

# make_pp <pred_root> <dataset> <model> <min_voxels>: write $PP_ROOT/<dataset>/<model>_ppN/fold_k.
make_pp() {
  local root=$1 dataset=$2 model=$3 k=$4 fold
  for fold in "${FOLDS[@]}"; do
    local src="${root}/${model}/fold_${fold}" dst="${PP_ROOT}/${dataset}/${model}_pp${k}/fold_${fold}"
    if [[ ! -d "${src}" ]]; then echo "WARNING: missing ${src}"; continue; fi
    echo "  creating ${dst}"
    "${PYTHON}" evaluation/postprocess_min_size.py --in_dir "${src}" --out_dir "${dst}" \
      --min_voxels "${k}" > /dev/null
  done
}

# run_eval <gt_dir> <mask_dir> <prob_dir> <out_dir> <name> <fold> <mode>
run_eval() {
  local gt=$1 mask=$2 prob=$3 out=$4 name=$5 fold=$6 mode=$7
  if [[ ! -d "${mask}" ]]; then echo "WARNING: missing ${mask}, skipped"; return; fi
  echo "  ${name} fold ${fold} (${mode})"
  mkdir -p "${out}"
  "${PYTHON}" evaluation/multiscale_eval.py \
    --gt_dir "${gt}" \
    --pred_mask_dir "${mask}" \
    --pred_prob_dir "${prob}" \
    --allow_missing_prob \
    --out_csv "${out}/multiscale_eval_${name}_fold${fold}.csv" \
    --out_json "${out}/multiscale_eval_${name}_fold${fold}.json" \
    --small_voxels_thresh 150 \
    --metrics_mode "${mode}" > /dev/null
}

# eval_folds <gt> <pred_root> <dataset> <out_dir> <mode> <name>... ; a name like CATMIL_pp5
# reads the masks from $PP_ROOT/<dataset>/CATMIL_pp5/ and the probabilities from CATMIL/.
# Output names follow results/per_fold: CATMIL_pp5 stays CATMIL_pp5, CATMIL_pp3 becomes CATMILpp3.
eval_folds() {
  local gt=$1 root=$2 dataset=$3 out=$4 mode=$5 name fold
  shift 5
  for name in "$@"; do
    local base="${name%%_pp*}" label="${name}" masks="${root}"
    if [[ "${name}" == *_pp* ]]; then masks="${PP_ROOT}/${dataset}"; fi
    if [[ "${name}" == *_pp* && "${name}" != *_pp5 ]]; then label="${name/_pp/pp}"; fi
    for fold in "${FOLDS[@]}"; do
      run_eval "${gt}" "${masks}/${name}/fold_${fold}" "${root}/${base}/fold_${fold}" \
        "${out}" "${label}" "${fold}" "${mode}"
    done
  done
}

mslesseg_main() {
  echo "MSLesSeg, main metrics"
  for m in CATMIL CCDiceCE FocalTversky Tversky nnUNet; do make_pp "${MS_PRED}" MSLesSeg "${m}" 5; done
  for k in 3 10 15; do make_pp "${MS_PRED}" MSLesSeg CATMIL "${k}"; done
  eval_folds "${MS_GT}" "${MS_PRED}" MSLesSeg "${OUT_ROOT}/MSLesSeg/main" main \
    CAT MIL CATMIL CCDiceCE FocalTversky Tversky nnUNet \
    CATMIL_pp5 CCDiceCE_pp5 FocalTversky_pp5 Tversky_pp5 nnUNet_pp5 \
    CATMIL_pp3 CATMIL_pp10 CATMIL_pp15
}

mslesseg_all() {
  echo "MSLesSeg, all metrics"
  for m in CATMIL FocalTversky Tversky nnUNet; do make_pp "${MS_PRED}" MSLesSeg "${m}" 5; done
  eval_folds "${MS_GT}" "${MS_PRED}" MSLesSeg "${OUT_ROOT}/MSLesSeg/all" all \
    CAT MIL CATMIL FocalTversky Tversky nnUNet UMambaBot \
    CATMIL_pp5 FocalTversky_pp5 Tversky_pp5 nnUNet_pp5
}

lambda() {
  echo "MSLesSeg, lambda sweep (one model per setting, fold_all)"
  local name dir
  for name in CAT01MIL01 CAT01MIL02 CAT02MIL01 CAT02MIL02; do
    dir="${MS_PRED}/${name}/CATMIL/fold_all"
    run_eval "${MS_GT}" "${dir}" "${dir}" "${OUT_ROOT}/MSLesSeg/lambda_sweep" "${name}" all all
  done
}

ljubljana() {
  echo "3D-MR-MS, all metrics"
  for m in CATMIL FocalTversky Tversky nnUNet; do make_pp "${LJ_PRED}" Ljubljana "${m}" 5; done
  eval_folds "${LJ_GT}" "${LJ_PRED}" Ljubljana "${OUT_ROOT}/Ljubljana/all" all \
    CATMIL CCDiceCE FocalTversky Tversky nnUNet \
    CATMIL_pp5 FocalTversky_pp5 Tversky_pp5 nnUNet_pp5
}

SETS=("$@")
if [[ ${#SETS[@]} -eq 0 ]]; then SETS=(mslesseg_main mslesseg_all lambda ljubljana); fi
for s in "${SETS[@]}"; do
  case "${s}" in
    mslesseg_main|mslesseg_all|lambda|ljubljana) "${s}" ;;
    *) echo "Unknown set: ${s}" >&2; exit 1 ;;
  esac
done
echo "Done. Results in ${OUT_ROOT}"
