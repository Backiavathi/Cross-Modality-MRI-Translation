"""
==============================================================================
Additional cells for DenseTransNet_Enhanced_subjectwise_classweighted_85_15.ipynb
==============================================================================
Addresses:
  - Reviewer Comment #6: Statistical rigor for Table 5
      * McNemar's test  (paired predictions, DenseTransNet vs each baseline)
      * DeLong's test   (AUC comparison, DenseTransNet vs each baseline)
      * 95% CI for AUC  (DeLong method, closed-form, no bootstrap needed)
  - Reviewer Comment #8: Quantitative Grad-CAM++ localization for Table 3
      * Dice and IoU of thresholded Grad-CAM++ heatmaps against
        atlas-based anatomical ROI masks (substantia nigra, basal ganglia)

Paste each numbered section into its own new cell, in order, AFTER Cell 13
(Grad-CAM++ Visualisation) in the DenseTransNet notebook. Both sections
assume `model`, `test_loader`, `DEVICE`, `test_df` already exist from
earlier cells.
==============================================================================
"""

# ==============================================================================
# CELL A — Install / imports for statistical testing
# ==============================================================================
# !pip install statsmodels -q

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torchvision import models as tv_models


# ==============================================================================
# CELL B — Train/load the five pre-trained baseline models
# ==============================================================================
# Each baseline is fine-tuned on the SAME train_loader/val_loader/test_loader
# used for DenseTransNet, so the comparison in Table 5 is apples-to-apples
# (same subject-level split, same class weights, same preprocessing).
#
# If you already have saved weights for these baselines from your original
# Table 5 run, skip training and just torch.load() them instead — the
# important part for reviewer comment #6 is the STATISTICAL TEST code in
# Cell D/E/F, not re-running these baselines from scratch.

def build_baseline(name, num_classes=2):
    if name == 'AlexNet':
        m = tv_models.alexnet(weights=tv_models.AlexNet_Weights.DEFAULT)
        m.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif name == 'InceptionV3':
        m = tv_models.inception_v3(weights=tv_models.Inception_V3_Weights.DEFAULT,
                                    aux_logits=True)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        m.AuxLogits.fc = torch.nn.Linear(m.AuxLogits.fc.in_features, num_classes)
    elif name == 'Xception':
        # torchvision has no native Xception; timm is the standard source.
        import timm
        m = timm.create_model('xception', pretrained=True, num_classes=num_classes)
    elif name == 'ResNet50':
        m = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    elif name == 'EfficientNetB7':
        m = tv_models.efficientnet_b7(weights=tv_models.EfficientNet_B7_Weights.DEFAULT)
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(name)
    return m.to(DEVICE)


def train_baseline(m, epochs=NUM_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY):
    cw = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        m.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = m(inputs)
            # InceptionV3 returns (logits, aux_logits) in train mode
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return m


@torch.no_grad()
def evaluate_baseline(m):
    m.eval()
    labels_all, preds_all, probs_all = [], [], []
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = m(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)
        labels_all.extend(labels.cpu().numpy())
        preds_all.extend(preds.cpu().numpy())
        probs_all.extend(probs[:, 1].cpu().numpy())
    return (np.array(labels_all), np.array(preds_all), np.array(probs_all))


baseline_names = ['AlexNet', 'InceptionV3', 'Xception', 'ResNet50', 'EfficientNetB7']
baseline_results = {}   # name -> (labels, preds, probs)

for name in baseline_names:
    print(f'Training {name} ...')
    bm = build_baseline(name)
    bm = train_baseline(bm)
    labels_b, preds_b, probs_b = evaluate_baseline(bm)
    baseline_results[name] = (labels_b, preds_b, probs_b)
    del bm
    torch.cuda.empty_cache()

# Proposed model's own test-set predictions (from Cell 11 in your notebook)
# all_labels, all_preds, all_probs already exist from Cell 11 — reused directly.
baseline_results['Dual-scale DenseTransNet'] = (
    np.array(all_labels), np.array(all_preds), np.array(all_probs)
)


# ==============================================================================
# CELL C — DeLong's test implementation
# ==============================================================================
# Standard fast DeLong algorithm (Sun & Xu, 2014) for comparing two
# correlated ROC AUCs computed on the SAME test set. No public
# scikit-learn/scipy function exists for this, so it is implemented
# directly (this is the standard reference implementation used in the
# medical-ML literature).

def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_transposed, label_1_count):
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    positive_examples = preds_sorted_transposed[:, :m]
    negative_examples = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, probs_a, probs_b):
    """
    Returns (auc_a, auc_b, z_statistic, two_sided_p_value)
    for two sets of predicted probabilities on the SAME y_true labels.
    """
    order = np.argsort(-y_true)
    y_true_sorted = y_true[order]
    label_1_count = int(np.sum(y_true_sorted))

    preds = np.vstack([probs_a[order], probs_b[order]])
    aucs, delongcov = _fast_delong(preds, label_1_count)

    l = np.array([[1, -1]])
    z = (l @ aucs) / np.sqrt(l @ delongcov @ l.T)
    p_value = 2 * (1 - stats.norm.cdf(abs(z[0])))
    return aucs[0], aucs[1], float(z[0]), float(p_value)


def delong_auc_ci(y_true, probs, alpha=0.95):
    """95% CI for a single model's AUC via the DeLong variance estimate."""
    order = np.argsort(-y_true)
    y_true_sorted = y_true[order]
    label_1_count = int(np.sum(y_true_sorted))
    preds = np.vstack([probs[order]])          # single-model "2-row trick"
    preds = np.vstack([preds, preds])           # duplicate row so _fast_delong runs
    aucs, delongcov = _fast_delong(preds, label_1_count)
    auc = aucs[0]
    var = delongcov[0, 0]
    se = np.sqrt(var)
    z_crit = stats.norm.ppf(1 - (1 - alpha) / 2)
    lower = max(0.0, auc - z_crit * se)
    upper = min(1.0, auc + z_crit * se)
    return auc, lower, upper


# ==============================================================================
# CELL D — McNemar's test (paired predictions vs proposed model)
# ==============================================================================

def mcnemar_test(y_true, preds_a, preds_b):
    """
    McNemar's test on paired binary predictions from two models on the
    same test set. Returns the p-value.
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    # contingency table:
    #            B correct   B incorrect
    # A correct     n11          n10
    # A incorrect   n01          n00
    n11 = np.sum(correct_a & correct_b)
    n10 = np.sum(correct_a & ~correct_b)
    n01 = np.sum(~correct_a & correct_b)
    n00 = np.sum(~correct_a & ~correct_b)

    table = np.array([[n11, n10], [n01, n00]])
    # exact binomial test is preferred for the small discordant-pair counts
    # typical of an n=65-subject independent test set
    result = mcnemar(table, exact=True)
    return result.pvalue


# ==============================================================================
# CELL E — Build the full Table 5 (Acc, Sp, Sn, AUC + 95% CI, McNemar p,
#           DeLong p, all relative to the proposed model)
# ==============================================================================

from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

proposed_labels, proposed_preds, proposed_probs = baseline_results['Dual-scale DenseTransNet']

rows = []
for name in baseline_names + ['Dual-scale DenseTransNet']:
    y_true, preds, probs = baseline_results[name]

    acc = accuracy_score(y_true, preds) * 100
    sn  = recall_score(y_true, preds) * 100
    sp  = specificity_score(y_true, preds) * 100
    auc, ci_lo, ci_hi = delong_auc_ci(y_true, probs)

    if name == 'Dual-scale DenseTransNet':
        mcnemar_p, delong_p = None, None
    else:
        mcnemar_p = mcnemar_test(y_true, preds, proposed_preds)
        _, _, _, delong_p = delong_roc_test(y_true, probs, proposed_probs)

    rows.append({
        'Model': name,
        'Acc (%)': round(acc, 2),
        'Sp (%)': round(sp, 2),
        'Sn (%)': round(sn, 2),
        'AUC': round(auc, 3),
        'AUC 95% CI': f'({ci_lo:.3f}\u2013{ci_hi:.3f})',
        'McNemar p': None if mcnemar_p is None else round(mcnemar_p, 3),
        'DeLong p': None if delong_p is None else round(delong_p, 3),
    })

table5_df = pd.DataFrame(rows)
print(table5_df.to_string(index=False))
table5_df.to_csv(os.path.join(OUTPUT_DIR if 'OUTPUT_DIR' in dir() else '.',
                               'table5_statistical_comparison.csv'), index=False)


# ==============================================================================
# CELL F — Grad-CAM++ Dice / IoU against anatomical ROI masks (Table 3)
# ==============================================================================
# Requires: a directory of BINARY ROI masks, one per test slice, registered
# to the SAME space/resolution as the classifier input (224x224), for each
# anatomical ROI (substantia nigra, basal ganglia). These come from your
# CAT12/atlas-based segmentation pipeline (e.g., an MNI atlas resampled and
# warped per-subject the same way your T1/T2 volumes were, then thresholded
# to a binary mask and saved as an image with the SAME filename stem as the
# corresponding test slice).
#
# Directory layout expected:
#   ROI_MASK_DIR/substantia_nigra/<slice_filename>
#   ROI_MASK_DIR/basal_ganglia/<slice_filename>

import cv2
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

ROI_MASK_DIR = "/path/to/roi_masks"       # <-- set this
ROI_NAMES = ['substantia_nigra', 'basal_ganglia']
CAM_THRESHOLD = 0.5   # heatmap values above this (normalised 0-1) -> foreground

target_layers = [model.late_features[-1]]   # same target layer as Cell 13
cam = GradCAMPlusPlus(model=model, target_layers=target_layers)


def dice_iou(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    dice = (2.0 * intersection) / (pred.sum() + gt.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    return dice, iou


results_per_roi = {roi: {'dice': [], 'iou': []} for roi in ROI_NAMES}

model.eval()
for idx in range(len(test_df)):
    row = test_df.iloc[idx]
    img_path = row['filepath']
    label = int(row['label'])

    # Load and preprocess exactly as in val_test_transform (Cell 4)
    pil_img = Image.open(img_path).convert('L')
    input_tensor = val_test_transform(pil_img).unsqueeze(0).to(DEVICE)

    targets = [ClassifierOutputTarget(label)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # (224,224), 0-1

    binary_cam = (grayscale_cam >= CAM_THRESHOLD).astype(np.uint8)

    for roi in ROI_NAMES:
        mask_path = os.path.join(ROI_MASK_DIR, roi, row['filename'])
        if not os.path.exists(mask_path):
            continue   # skip slices without a matching ROI mask
        gt_mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask_img = cv2.resize(gt_mask_img, (224, 224),
                                  interpolation=cv2.INTER_NEAREST)
        gt_binary = (gt_mask_img > 127).astype(np.uint8)

        d, i = dice_iou(binary_cam, gt_binary)
        results_per_roi[roi]['dice'].append(d)
        results_per_roi[roi]['iou'].append(i)

# ---- Summarise into Table 3 ----
table3_rows = []
all_dice, all_iou = [], []
for roi in ROI_NAMES:
    dice_vals = np.array(results_per_roi[roi]['dice'])
    iou_vals = np.array(results_per_roi[roi]['iou'])
    all_dice.extend(dice_vals)
    all_iou.extend(iou_vals)
    table3_rows.append({
        'Anatomical ROI': roi.replace('_', ' ').title(),
        'No. of Subjects': len(dice_vals),
        'Dice (mean \u00b1 SD)': f'{dice_vals.mean():.2f} \u00b1 {dice_vals.std():.2f}',
        'IoU (mean \u00b1 SD)': f'{iou_vals.mean():.2f} \u00b1 {iou_vals.std():.2f}',
    })

all_dice = np.array(all_dice)
all_iou = np.array(all_iou)
table3_rows.append({
    'Anatomical ROI': 'Overall',
    'No. of Subjects': len(all_dice),
    'Dice (mean \u00b1 SD)': f'{all_dice.mean():.2f} \u00b1 {all_dice.std():.2f}',
    'IoU (mean \u00b1 SD)': f'{all_iou.mean():.2f} \u00b1 {all_iou.std():.2f}',
})

table3_df = pd.DataFrame(table3_rows)
print(table3_df.to_string(index=False))
table3_df.to_csv(os.path.join(OUTPUT_DIR if 'OUTPUT_DIR' in dir() else '.',
                               'table3_gradcam_dice_iou.csv'), index=False)
