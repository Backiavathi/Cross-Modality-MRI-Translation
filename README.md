# Cross-Modality-MRI-Translation
Official implementation of a CycleGAN-based T1-to-T2 MRI translation framework with Dual-scale DenseTransNet for Parkinson’s disease classification using MRI.
## Overview

This repository contains the implementation associated with our study on Parkinson's disease classification using T1- and T2-weighted MRI data.

The framework consists of two major components:

1. **CycleGAN-based T1-to-T2 MRI translation**
   - Generates synthetic T2-weighted MRI images from T1-weighted MRI images.
   - Uses adversarial and cycle-consistency learning.
   - An auxiliary SSIM loss is used during training when paired T2-weighted reference images are available.

2. **Dual-scale DenseTransNet**
   - Performs Parkinson's disease classification using MRI images.
   - Combines convolutional and transformer-based feature representation.
   - Performance is evaluated using accuracy, specificity, sensitivity, precision, F1-score, and ROC-AUC.

## Repository Structure

```text
CycleGAN-and-Dual-scale-DenseTransNet-Parkinson-MRI/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── README.md

├── cyclegan/
│   ├── train_cyclegan.py
│   ├── generator.py
│   ├── discriminator.py
│   └── losses.py
│
├── classification/
│   ├── train_dense_transnet.py
│   ├── model.py
│   ├── dataset.py
│   └── evaluation.py
│
├── explainability/
│   ├── gradcam_pp.py
│   └── roi_overlap.py
│
└── statistical_analysis/
    ├── delong_test.py
    ├── mcnemar_test.py
    └── confidence_intervals.py
```
## Dataset

The MRI data used in this study were obtained from the Parkinson's Progression Markers Initiative (PPMI) database.

The study cohort consists of 430 subjects, including 270 subjects with Parkinson's disease (PD) and 160 healthy controls (HC). Ten axial 2D slices were extracted from each subject, resulting in a total of 4,300 slices across both classes.

The PPMI MRI data are not redistributed in this repository. Qualified researchers can obtain the data through the PPMI data-access procedures and applicable data-use agreement.

## Subject-wise Splitting

To prevent data leakage, dataset partitioning was performed at the subject level before slice extraction.

A total of 365 subjects (3,650 slices) were used for training/development and 65 independent subjects (650 slices) were reserved for final testing.

For 10-fold cross-validation, StratifiedGroupKFold was used with subject ID as the grouping variable. Thus, all slices belonging to the same subject remained within the same fold.

## CycleGAN-based T1-to-T2 Translation

The first stage of the framework performs T1-to-T2 MRI translation using CycleGAN.

The model uses adversarial, cycle-consistency, and identity losses. Because paired T1- and T2-weighted MRI scans are available for the selected subjects, an auxiliary SSIM loss is incorporated during training to improve structural fidelity.

The SSIM loss is defined as:

L_SSIM = 1 - SSIM(y_real, y_fake)

where y_real denotes the paired real T2-weighted image and y_fake denotes the synthesized T2-weighted image.

During inference, only the trained T1-to-T2 generator is required; a paired T2-weighted image is not required.

## Dual-scale DenseTransNet

The second stage uses the proposed Dual-scale DenseTransNet for Parkinson's disease classification.

The framework combines convolutional and transformer-based feature representations to capture complementary local and global information from MRI images.

The classifier uses a 3 × 224 × 224 input representation, where the three channels correspond to the replicated grayscale MRI input.

## Evaluation Metrics

Classification performance is evaluated using:

- Accuracy
- Specificity
- Sensitivity
- Precision
- F1-score
- ROC-AUC

Image-generation quality is evaluated using:

- SSIM
- PSNR

## Grad-CAM++ Explainability

Grad-CAM++ is used to visualize image regions contributing to the model's predictions.

Quantitative localization analysis is performed using anatomical atlas-based regions of interest (ROIs), including the substantia nigra and basal ganglia.

Localization performance is evaluated using:

- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU)

These quantitative measures complement the qualitative Grad-CAM++ visualizations.

## Statistical Analysis

Statistical analysis includes:

- 95% confidence intervals for ROC-AUC
- DeLong's test for correlated ROC-AUC comparisons
- McNemar's test for paired classification outcomes
- Mean ± standard deviation for 10-fold cross-validation

Statistical comparisons are performed using subject-level predictions when the required paired predictions are available.

Published methods are not statistically compared using subject-level tests when their individual predictions are unavailable.

## Reproducibility

To reproduce the experiments:

1. Obtain the required MRI data through the PPMI data-access procedure.
2. Perform the preprocessing described in the manuscript.
3. Create the subject-level training/development and independent test split.
4. Extract 10 axial slices per subject.
5. Train the CycleGAN-based T1-to-T2 translation model.
6. Generate synthetic T2-weighted images.
7. Train Dual-scale DenseTransNet.
8. Perform subject-level 10-fold cross-validation using StratifiedGroupKFold.
9. Evaluate the final model on the independent test subjects.
10. Generate Grad-CAM++ visualizations.
11. Perform anatomical ROI Dice/IoU analysis.
12. Perform the statistical analyses.

## Code Availability

The source code and implementation details of the proposed CycleGAN and Dual-scale DenseTransNet framework are provided in this repository to facilitate reproducibility.

The PPMI MRI data are not included in this repository because the original data are subject to the PPMI data-access and data-use requirements.

A total of 365 subjects (3,650 slices) were used for training/development and 65 independent subjects (650 slices) were reserved for final testing.

For 10-fold cross-validation, StratifiedGroupKFold was used with subject ID as the grouping variable. Thus, all slices belonging to the same subject remained within the same fold.
