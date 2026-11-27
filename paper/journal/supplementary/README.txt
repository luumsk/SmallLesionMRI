Supplementary Materials: per-case and per-lesion results
=======================================================

Component-Adaptive and Lesion-Level Supervision for Improved Small Structure Segmentation
in Brain MRI. Code: https://github.com/luumsk/SmallLesionMRI (analysis/export_supplementary.py
writes these files from the evaluation outputs in results/).

The files contain derived metrics only; no images or masks. MSLesSeg case IDs are
<patient>_<timepoint> (for example P13_T1); 3D-MR-MS case IDs are the dataset's patient IDs.
Lesions are 6-connected components of the reference mask; a small lesion has at most
150 voxels in either dataset (MSLesSeg voxels are 1 mm^3; 3D-MR-MS voxels are about
0.18 mm^3, so size_mm3 differs from size_vox there). A reference lesion counts as detected if the
predicted mask (foreground probability > 0.5) overlaps it by at least one voxel.
Losses: CATMIL (ours), DiceCE, Tversky, FocalTversky, CC-DiceCE; CAT and MIL are the
ablations with one of the two terms. fold = the cross-validation fold model (0-4) that made
the prediction. Paper tables report the mean over cases within each fold, then the mean and
sample standard deviation over the five folds. Values are rounded to 6 decimals.
Empty cells: metric undefined (for example small-lesion recall for a case without small lesions).

TableS1_test_per_case.csv: one row per dataset, loss, post-processing, fold and test case
  dataset                    MSLesSeg (12 test scans) or 3D-MR-MS (6 test scans, external)
  loss                       training loss (nnU-Net)
  post_processing            none, or PPs = predicted components smaller than s voxels removed
  fold, case_id              see above
  n_reference_lesions        reference lesions in the case
  n_small_reference_lesions  of which small
  small_lesion_recall        fraction of small reference lesions detected
  fn_lesion_count            reference lesions missed (all sizes)
  fn_volume_fraction         volume of missed reference lesions / total reference lesion volume
  dice                       voxel-wise Dice similarity coefficient
  hd95_mm                    95th-percentile Hausdorff distance (mm)
  lesion_f1                  lesion-wise F1 (harmonic mean of lesion-wise precision and recall)
  fp_volume_mm3              volume of predicted voxels outside the reference mask

TableS2_test_per_lesion.csv: one row per reference lesion, loss and fold, no post-processing
  dataset, loss, fold, case_id   as in Table S1
  lesion_id                      lesion number within the case's reference mask
  size_vox, size_mm3             lesion size
  longest_inplane_diameter_mm    longest diameter within any axial, coronal or sagittal slice
                                 (edge to edge; the measure behind the 3 mm clinical criterion)
  longest_3d_diameter_mm         longest 3D (Feret) diameter, edge to edge
  small                          size_vox <= 150
  detected                       True if detected (see above)
  coverage                       fraction of the lesion's voxels inside the predicted mask
  max_probability                highest predicted foreground probability inside the lesion

TableS3_cv_per_scan.csv: one row per split, model, fold and scan
  split                     cross-validation: each of the 81 MSLesSeg training scans predicted by
                            the fold model that did not train on it; test: the MSLesSeg test scans
                            predicted by each fold model
  architecture, loss        network and training loss
  fold, patient, case_id    fold model, patient ID, scan ID
  n_reference_lesions       reference lesions in the scan
  dice, small_lesion_recall, fn_lesion_count, lesion_f1   as in Table S1
  *_pp5                     the same metrics after removing predicted components under 5 voxels
