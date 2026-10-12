# Presentation assets

## Files

- `make_assets.py`
- `clean_normal.png`
- `clean_pneumonia.png`
- `case_A_input.png`
- `case_A_phase1_cam.png`
- `case_A_phase2_gradcam.png`
- `case_B_input.png`
- `case_B_phase1_cam.png`
- `case_B_phase2_gradcam.png`
- `case_C_input.png`
- `case_C_phase1_cam.png`
- `case_C_phase2_gradcam.png`
- `cases_summary.csv`
- `confusion_matrix.png`
- `roc_curve.png`
- `classification_check.json`
- `README.md`

## Image IDs and case selection

- Clean Normal image (label `Normal`, test split): `0022995a-45eb-4cfa-9a59-cd15f5196c64`.
- Clean Pneumonia image (label `Lung Opacity`, test split): `aa720b92-39d2-4db4-a6c1-318840cb973b`.
- A_success: `aa720b92-39d2-4db4-a6c1-318840cb973b`; Phase 1 IoU 0.773519; Phase 2 IoU 0.717053.
- B_partial: `a5df1553-7c85-46e6-9e0b-870d67d23eb7`; Phase 1 IoU 0.001224; Phase 2 IoU 0.054689.
- C_failure: `77a4a76f-d348-4dd4-b22a-d7b676121d78`; Phase 1 IoU 0.000000; Phase 2 IoU 0.000000.

The 12 candidate IDs came from the existing Stage 12 failure-analysis index in `results/phase2_error_analysis/examples.csv`; its indexed IoUs are Phase 1 baseline values, verified against `results/cam_results.json`. Phase 2 maps and per-image IoUs were then generated for those 12 candidates with the saved method. Cases are the highest Phase 2 IoU, the candidate nearest the saved Phase 2 full-test median (0.068268), and the lowest Phase 2 IoU. The Grad-CAM++ implementation was verified against a saved Stage 13 example before this selection. Phase 1 maps and both sets of boxes were produced only for the three selected test images, using the unchanged fixed split and epoch-9 checkpoint.

## Production and assumptions

- Source pixels were decoded with `src.data.preprocessing.read_dicom_pixels`; the clean PNGs are independently min-max windowed to the full stored pixel range (after the DICOM MONOCHROME1 handling), converted to 8-bit grayscale, and resized without added overlays or text. Any orientation/acquisition markers already encoded in the source pixels remain. The clean files preserve aspect ratio with their long side at 1024 px.
- CNN inputs use the existing train-fitted preprocessing function and statistics. Phase 1 maps use `src.localization.cam.compute_cam`; Phase 2 maps use `src.localization.improved_cam.compute_gradcam_plus_plus`. Both use class 1, inclusive threshold 0.50, iterative 8-connected clustering, and the existing 2-SD region filter and box conversion.
- Ground-truth boxes come from each test row in the unchanged `metadata/split_metadata.csv`. `src.localization.iou.image_iou` is used for greedy one-to-one IoU matching. DICOM-native box coordinates are transformed through one shared aspect-preserving square fit for each case image and both overlays.
- Result overlays use a JET heatmap at 0.40 opacity, ground-truth green `#00C853`, predicted orange `#FF6D00`, and 4 px outlines. Square case images are padded, not cropped, so all layers share the same scale and placement.
- Classification was inference only on all 1,486 test images. Metrics passed the sanity check against `results/final_model_comparison.csv`: accuracy 0.880215343; precision 0.920477137; recall 0.770382696; F1 0.838768116. The ROC-AUC and confusion counts in the JSON were computed from this inference.
- No model was retrained, no split was generated, and files in `results/`, `metadata/`, and `notebooks/` were not modified. No external packages were installed.

