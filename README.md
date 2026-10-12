# Weakly Supervised Pneumonia Localization

## Overview

This project reproduces a reference-paper-inspired pneumonia classification and weakly supervised localization workflow on the available RSNA Pneumonia Detection Challenge Stage 2 data. It evaluates classical classifiers, a ten-layer CNN, and CNN class activation maps (CAM). A second phase analyzes CAM failures and evaluates Grad-CAM++ localization using the same frozen CNN.

The local cohort and some implementation details differ from the paper. Results are reported as a reproduction attempt, not as an exact replication. The RSNA image dataset is **not included in this repository**; obtain it separately under its applicable terms.

## Reference Paper

The reference is identified in the project as **“Weakly Supervised Pneumonia Localization.”** Verified author, venue, and DOI information is not present in the project records and is therefore not inferred.

## Key Contributions

### Reference-inspired reproduction

- Logistic Regression (LR)
- Linear Support Vector Machine (SVM)
- Random Forest (RF)
- Paper-inspired 10-layer CNN
- Baseline classifier-weighted CAM localization

### Extensions

- **XGBoost:** PCA with 256 components, fitted on training data only, followed by a fixed XGBoost classifier. This was not part of the reference method.
- **Phase 2 localization:** Stage 12 CAM failure and threshold analysis, Stage 13 Grad-CAM++ evaluation with the frozen CNN, and Stage 14 ablation. The map generator changed; the Phase 1 threshold, region post-processing, box scaling, fixed test split, and IoU matching were kept the same.

See [RESULTS.md](./RESULTS.md) for Phase 1 and classification results and [PHASE2_RESULTS.md](./PHASE2_RESULTS.md) for the Phase 2 comparison.

## Pipeline

```text
RSNA Stage 2 X-rays and labels (local; not committed)
                    |
                    v
      Metadata + fixed stratified split
                    |
                    v
       DICOM resize and normalization
                    |
          +---------+----------+
          |                    |
          v                    v
  Flattened pixels       10-layer CNN
          |                    |
    +-----+------+             v
    |     |      |       Phase 1 CAM
    LR   SVM     RF            |
                              v
                      Phase 2 Grad-CAM++
                              |
                              v
                   Box localization and IoU

Separate classification extension:
Preprocessed images -> flatten -> train-only PCA (256) -> XGBoost
```

## Dataset

The project uses the locally available RSNA Stage 2 training labels and images. The usable binary cohort contains **14,863 labeled images**: 6,012 Pneumonia (`Lung Opacity`) and 8,851 Healthy (`Normal`). The other labeled class is excluded. The fixed seed-42 stratified 70/20/10 assignment is stored in `metadata/split_metadata.csv` (10,405 train, 2,972 validation, 1,486 test). CAM localization scores the 601 Pneumonia-positive images in that test split.

The reference project notes a 17,489-image binary cohort. The local cohort is smaller and differs in class counts, so the comparison is not cohort matched. **Do not commit the RSNA dataset.** Place the required local files as follows:

```text
Data/
├── stage_2_train_images/
├── stage_2_train_labels.csv
└── stage_2_detailed_class_info.csv
```

`Data/` is ignored by Git. See [metadata/DATASET_STATS.md](./metadata/DATASET_STATS.md) and [metadata/SPLIT_STATS.md](./metadata/SPLIT_STATS.md) for recorded counts.

## Installation

From the repository root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The saved Stage 10 result records Python 3.14.4. The requirements file uses minimum versions and is not a fully locked environment.

## Running the Project

### Quick Phase 1 + Phase 2 demo

To run both localization methods on one Pneumonia-positive image from the
existing fixed test split, using the saved epoch-9 CNN checkpoint, run:

```powershell
python main.py
```

The demo is inference-only: it does not retrain a model, regenerate the split,
or run full-test evaluation. It saves the input X-ray, Phase 1 CAM overlay,
Phase 2 Grad-CAM++ overlay, and a JSON summary under a new timestamped folder
in `demo_outputs/`. You can run one phase at a time or choose a test image:

```powershell
python main.py --phase phase1
python main.py --phase phase2
python main.py --phase both --image-id 0087bd3a-55a7-4045-b111-b018fa52d361
```

The selected image ID must be Pneumonia-positive and assigned to the existing
test split. Green boxes show ground truth; orange boxes show model predictions.
This is an educational visualization, not a clinical diagnosis.

The following commands correspond to existing scripts. Dataset creation and training commands can overwrite outputs or take substantial time. **Do not run them just to review the committed/saved results.** In particular, do not recreate the split when reproducing metrics against the saved artifacts.

```powershell
# Stage 1 dataset metadata, fixed split, and preprocessing verification
python scripts/create_metadata.py
python scripts/create_split.py
python scripts/verify_preprocessing.py --samples 6

# Phase 1 classification models (training)
python scripts/train_logistic_regression.py
python scripts/train_svm.py
python scripts/train_random_forest.py
python scripts/train_cnn.py

# Phase 1 CAM full test evaluation (uses the existing trained CNN)
python scripts/run_cam_localization.py

# XGBoost extension: small smoke check, then full training run
python scripts/train_xgboost.py --smoke-test
python scripts/train_xgboost.py

# Phase 2 Stage 12 analysis and Stage 13 smoke/full evaluation
python scripts/analyze_cam_errors.py
python scripts/run_improved_cam.py --smoke-only --threshold 0.5 --batch-size 12
python scripts/run_improved_cam.py --threshold 0.5 --batch-size 16

# Stage 14 reuses the saved Stage 9 and Stage 13 results; it does not rerun CAM
python scripts/summarize_phase2_ablation.py
```

Stage 15 final comparison and Stage 16 finalization are documentation-only; their outputs are the saved comparison files, notebooks, and reports. The Phase 2 analysis/evaluation commands above process the fixed test set and are not needed when the saved result files are present. Full reproduction guidance and safeguards are in [REPRODUCTION.md](./REPRODUCTION.md).

## Notebooks

All nine numbered notebooks are present under `notebooks/`.

1. [01_Dataset_Preprocessing.ipynb](./notebooks/01_Dataset_Preprocessing.ipynb) — documents existing metadata and preprocessing outputs.
2. [02_Baseline_Models.ipynb](./notebooks/02_Baseline_Models.ipynb) — LR, SVM, and RF baseline results.
3. [03_CNN_Reproduction.ipynb](./notebooks/03_CNN_Reproduction.ipynb) — paper-inspired CNN architecture, training, and evaluation.
4. [04_CAM_Localization.ipynb](./notebooks/04_CAM_Localization.ipynb) — Phase 1 CAM generation, post-processing, and evaluation.
5. [05_XGBoost_Extension.ipynb](./notebooks/05_XGBoost_Extension.ipynb) — train-only PCA and XGBoost extension.
6. [06_CAM_Error_Analysis.ipynb](./notebooks/06_CAM_Error_Analysis.ipynb) — Stage 12 failure analysis and threshold sensitivity.
7. [07_Improved_CAM.ipynb](./notebooks/07_Improved_CAM.ipynb) — Stage 13 Grad-CAM++ comparison.
8. [08_Phase2_Ablation.ipynb](./notebooks/08_Phase2_Ablation.ipynb) — Stage 14 ablation findings.
9. [09_Phase2_Final_Comparison.ipynb](./notebooks/09_Phase2_Final_Comparison.ipynb) — Stage 15 final quantitative comparison.

Some notebooks contain training or full-test evaluation cells. Avoid executing them when reviewing saved results.

## Repository Structure

```text
.
├── Data/                         # Local RSNA dataset; ignored, not committed
├── metadata/                     # Dataset/split metadata and train-fitted stats
├── models/                       # Local CNN checkpoint when available; ignored
├── notebooks/                    # Nine numbered workflow/result notebooks
├── results/                      # Saved metrics, models, tables, selected examples
├── scripts/                      # Metadata, model, CAM, and summary entry points
├── src/
│   ├── data/                     # DICOM preprocessing and fixed-split dataset
│   ├── localization/             # Baseline CAM, Grad-CAM++, boxes, IoU
│   └── models/                   # Classical, CNN, and XGBoost models
├── FINAL_REPORT.md
├── PHASE2_PLAN.md
├── PHASE2_RESULTS.md
├── PROJECT_STATUS.md
├── PROJECT_PLAN.md
├── RESULTS.md
├── REPRODUCTION.md
└── requirements.txt
```

## Results

- [Phase 1 classification and localization results](./RESULTS.md)
- [Phase 2 error analysis, improved localization, ablation, and final comparison](./PHASE2_RESULTS.md)
- [Final project report](./FINAL_REPORT.md)
- [Detailed project documentation and manuscript draft](./PROJECT_DOCUMENTATION.md)
- [Project status](./PROJECT_STATUS.md)
- [Phase 2 final localization table](./results/phase2_final_comparison.csv)

Phase 1 CAM mean IoU is **0.080087**. Phase 2 Grad-CAM++ mean IoU is **0.090277** on the same 601-image test set (+0.010190, +12.72%). The paper-reported mean IoU is **0.1508**; Phase 2 remains below it. The IoU ≥ 0.5 rate remains 0.33%.

## Reproduction vs. Extension

The paper-inspired components are LR, SVM, RF, the 10-layer CNN, and baseline CAM localization. Implementation details, dataset counts, and results differ from or are not fully specified by the reference record; this project does not claim exact replication.

XGBoost with train-only PCA and Phase 2 Grad-CAM++ localization are extensions. Grad-CAM++ replaces only the Phase 1 map-generation method; the CNN is frozen, no bounding-box annotations enter prediction or tuning, and the original Phase 1 implementation/results are preserved.
