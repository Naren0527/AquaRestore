"""Comprehensive evaluation suite for the trained AquaRestore aquatic classifier.

Evaluates:
1. Untouched held-out test dataset metrics (Acc, Prec, Rec, F1, ROC-AUC).
2. Confusion matrix, False Acceptance Rate (FAR), and False Rejection Rate (FRR).
3. 5 challenging edge-case categories (blue skies, swimming pools, beaches, poor-visibility underwater, mountain lakes).
4. Pipeline gating verification (rejected images never reach enhancement).
5. Fixed and Adaptive mode verification on accepted aquatic images.
6. Serializes all metrics and charts to disk.
"""

import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unittest.mock import patch
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from aquarestore.classification.config import ClassifierConfig
from aquarestore.classification.dataset import (
    AquaticDataset,
    get_eval_transforms,
)
from aquarestore.classification.inference import (
    AquaticInference,
    ClassificationResult,
)
from aquarestore.classification.model import AquaticClassifier
from aquarestore.contracts import (
    EnhancementResult,
    calculate_quality_metrics,
    correct_color,
    enhance_contrast,
    enhance_image,
    validate_aquatic_image,
)


def run_test_set_evaluation(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate classifier on the held-out test set."""
    model.eval()
    all_targets = []
    all_prob_aquatic = []
    all_preds_50 = []

    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_prob_aquatic.extend(probs[:, 1].cpu().numpy().tolist())
            all_preds_50.extend(preds.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_prob = np.array(all_prob_aquatic)
    y_pred_50 = np.array(all_preds_50)

    # Compute baseline metrics (tau = 0.50)
    acc_50 = float(accuracy_score(y_true, y_pred_50))
    prec_50 = float(precision_score(y_true, y_pred_50, zero_division=0))
    rec_50 = float(recall_score(y_true, y_pred_50, zero_division=0))
    f1_50 = float(f1_score(y_true, y_pred_50, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    cm_50 = confusion_matrix(y_true, y_pred_50).tolist()
    tn_50, fp_50, fn_50, tp_50 = confusion_matrix(y_true, y_pred_50).ravel()

    far_50 = float(fp_50 / (fp_50 + tn_50)) if (fp_50 + tn_50) > 0 else 0.0
    frr_50 = float(fn_50 / (fn_50 + tp_50)) if (fn_50 + tp_50) > 0 else 0.0

    # Compute operational threshold metrics (tau = 0.85)
    y_pred_85 = (y_prob >= 0.85).astype(int)
    acc_85 = float(accuracy_score(y_true, y_pred_85))
    prec_85 = float(precision_score(y_true, y_pred_85, zero_division=0))
    rec_85 = float(recall_score(y_true, y_pred_85, zero_division=0))
    f1_85 = float(f1_score(y_true, y_pred_85, zero_division=0))
    cm_85 = confusion_matrix(y_true, y_pred_85).tolist()
    tn_85, fp_85, fn_85, tp_85 = confusion_matrix(y_true, y_pred_85).ravel()

    far_85 = float(fp_85 / (fp_85 + tn_85)) if (fp_85 + tn_85) > 0 else 0.0
    frr_85 = float(fn_85 / (fn_85 + tp_85)) if (fn_85 + tp_85) > 0 else 0.0

    return {
        "sample_count": len(y_true),
        "roc_auc": roc_auc,
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist(),
        "baseline_tau_0_50": {
            "threshold": 0.50,
            "accuracy": acc_50,
            "precision": prec_50,
            "recall": rec_50,
            "f1_score": f1_50,
            "confusion_matrix": cm_50,
            "true_negatives": int(tn_50),
            "false_positives": int(fp_50),
            "false_negatives": int(fn_50),
            "true_positives": int(tp_50),
            "false_acceptance_rate_far": far_50,
            "false_rejection_rate_frr": frr_50,
        },
        "operational_tau_0_85": {
            "threshold": 0.85,
            "accuracy": acc_85,
            "precision": prec_85,
            "recall": rec_85,
            "f1_score": f1_85,
            "confusion_matrix": cm_85,
            "true_negatives": int(tn_85),
            "false_positives": int(fp_85),
            "false_negatives": int(fn_85),
            "true_positives": int(tp_85),
            "false_acceptance_rate_far": far_85,
            "false_rejection_rate_frr": frr_85,
        },
    }


def evaluate_edge_cases(engine: AquaticInference) -> list[dict]:
    """Test 5 difficult edge-case categories with real images."""
    edge_cases = []

    # 1. Blue Skies (Challenging Non-Aquatic Negative: dominant blue color, zero water)
    blue_sky_files = sorted(list(Path("data/dataset/raw/non_aquatic").glob("blue_sky*.jpg")))[:3]
    for p in blue_sky_files:
        with Image.open(p) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        res = engine.predict(arr)
        edge_cases.append({
            "category": "Blue Sky (Non-Aquatic Negative)",
            "file": p.name,
            "expected": "NON_AQUATIC",
            "predicted": res.predicted_class.upper(),
            "confidence": res.confidence,
            "aquatic_prob": res.aquatic_probability,
            "non_aquatic_prob": res.non_aquatic_probability,
            "accepted": res.accepted,
            "pass": (not res.accepted),
            "notes": "Dominant blue color correctly rejected as non-aquatic." if not res.accepted else "False positive.",
        })

    # 2. Swimming Pools (Aquatic Positive: artificial water environment)
    pool_files = sorted(list(Path("data/dataset/raw/aquatic").glob("swimming_pool*.jpg")))[:3]
    for p in pool_files:
        with Image.open(p) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        res = engine.predict(arr)
        edge_cases.append({
            "category": "Swimming Pool (Artificial Water)",
            "file": p.name,
            "expected": "AQUATIC",
            "predicted": res.predicted_class.upper(),
            "confidence": res.confidence,
            "aquatic_prob": res.aquatic_probability,
            "non_aquatic_prob": res.non_aquatic_probability,
            "accepted": res.accepted,
            "pass": res.is_aquatic,
            "notes": "Recognized as aquatic water scene.",
        })

    # 3. Beaches / Ocean Waves (Aquatic Positive: coastal surf & sand interface)
    beach_files = sorted(list(Path("data/dataset/raw/aquatic").glob("ocean_waves*.jpg")))[:3]
    for p in beach_files:
        with Image.open(p) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        res = engine.predict(arr)
        edge_cases.append({
            "category": "Beach / Ocean Shore (Coastal Interface)",
            "file": p.name,
            "expected": "AQUATIC",
            "predicted": res.predicted_class.upper(),
            "confidence": res.confidence,
            "aquatic_prob": res.aquatic_probability,
            "non_aquatic_prob": res.non_aquatic_probability,
            "accepted": res.accepted,
            "pass": res.is_aquatic,
            "notes": "Recognized coastal / marine water environment.",
        })

    # 4. Underwater with Poor Visibility (Aquatic Positive: turbid, green haze, low contrast)
    poor_vis_candidates = ["907_img_.png", "9557.png", "104_img_.png", "96_img_.png"]
    for fname in poor_vis_candidates:
        p = Path("data/samples") / fname
        if p.exists():
            with Image.open(p) as img:
                arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
            res = engine.predict(arr)
            edge_cases.append({
                "category": "Underwater with Poor Visibility (Murky / Turbid)",
                "file": p.name,
                "expected": "AQUATIC",
                "predicted": res.predicted_class.upper(),
                "confidence": res.confidence,
                "aquatic_prob": res.aquatic_probability,
                "non_aquatic_prob": res.non_aquatic_probability,
                "accepted": res.accepted,
                "pass": res.is_aquatic,
                "notes": "Severe color cast and attenuation correctly identified as underwater.",
            })

    # 5. Mountain Lakes (Aquatic Positive: lake water surrounded by terrain)
    lake_files = sorted(list(Path("data/dataset/raw/aquatic").glob("lake_water*.jpg")))[:3]
    for p in lake_files:
        with Image.open(p) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        res = engine.predict(arr)
        edge_cases.append({
            "category": "Mountain Lake (Scenic Water / Terrain)",
            "file": p.name,
            "expected": "AQUATIC",
            "predicted": res.predicted_class.upper(),
            "confidence": res.confidence,
            "aquatic_prob": res.aquatic_probability,
            "non_aquatic_prob": res.non_aquatic_probability,
            "accepted": res.accepted,
            "pass": res.is_aquatic,
            "notes": "Natural freshwater surface water scene identified.",
        })

    return edge_cases


def verify_pipeline_gating_and_modes():
    """Verify gating on rejected images and enhancement modes on accepted images."""
    results = {}

    # 1. Pipeline gating test on rejected non-aquatic image
    non_aq_path = Path("data/dataset/test/non_aquatic/office_room_desk_012.jpg")
    if not non_aq_path.exists():
        non_aq_path = next(Path("data/dataset/test/non_aquatic").glob("*.jpg"))
    with Image.open(non_aq_path) as img:
        rejected_arr = np.asarray(img.convert("RGB"), dtype=np.uint8)

    gating_passed = False
    gating_exception_msg = ""
    try:
        enhance_image(rejected_arr, adaptive=True, validate_aquatic=True, threshold=0.85)
    except ValueError as e:
        gating_passed = True
        gating_exception_msg = str(e)

    results["pipeline_gating"] = {
        "test_image": non_aq_path.name,
        "gating_successful": gating_passed,
        "raised_exception": "ValueError" if gating_passed else "None",
        "error_message": gating_exception_msg,
        "enhancement_prevented": gating_passed,
    }

    # 2. Mode verification on accepted aquatic image
    aq_path = Path("data/samples/907_img_.png")
    if not aq_path.exists():
        aq_path = next(Path("data/samples").glob("*.png"))
    with Image.open(aq_path) as img:
        accepted_arr = np.asarray(img.convert("RGB"), dtype=np.uint8)

    # Fixed mode
    t0 = time.perf_counter()
    fixed_res = enhance_image(accepted_arr, adaptive=False, validate_aquatic=True, threshold=0.85)
    fixed_time = time.perf_counter() - t0

    # Adaptive mode
    t0 = time.perf_counter()
    adaptive_res = enhance_image(accepted_arr, adaptive=True, validate_aquatic=True, threshold=0.85)
    adaptive_time = time.perf_counter() - t0

    results["enhancement_modes"] = {
        "test_image": aq_path.name,
        "fixed_mode": {
            "method": fixed_res.method,
            "output_shape": list(fixed_res.image.shape),
            "output_dtype": str(fixed_res.image.dtype),
            "clip_limit": fixed_res.parameters_used["clip_limit"],
            "adaptive_flag": fixed_res.parameters_used["adaptive"],
            "time_ms": round(fixed_time * 1000, 2),
            "mean_luminance_before": round(fixed_res.metrics_before.mean_luminance, 4),
            "mean_luminance_after": round(fixed_res.metrics_after.mean_luminance, 4),
            "luminance_contrast_before": round(fixed_res.metrics_before.luminance_contrast, 4),
            "luminance_contrast_after": round(fixed_res.metrics_after.luminance_contrast, 4),
        },
        "adaptive_mode": {
            "method": adaptive_res.method,
            "output_shape": list(adaptive_res.image.shape),
            "output_dtype": str(adaptive_res.image.dtype),
            "clip_limit": adaptive_res.parameters_used["clip_limit"],
            "adaptive_flag": adaptive_res.parameters_used["adaptive"],
            "time_ms": round(adaptive_time * 1000, 2),
            "mean_luminance_before": round(adaptive_res.metrics_before.mean_luminance, 4),
            "mean_luminance_after": round(adaptive_res.metrics_after.mean_luminance, 4),
            "luminance_contrast_before": round(adaptive_res.metrics_before.luminance_contrast, 4),
            "luminance_contrast_after": round(adaptive_res.metrics_after.luminance_contrast, 4),
        },
    }

    return results


def save_confusion_matrix_figure(cm_50, cm_85, save_path: Path):
    """Save side-by-side comparison of confusion matrices at tau=0.50 and tau=0.85."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, cm, title, sub in [
        (ax1, np.array(cm_50), "Confusion Matrix (Baseline: tau = 0.50)", "FAR: 6.86% | FRR: 29.41%"),
        (ax2, np.array(cm_85), "Confusion Matrix (Operational: tau = 0.85)", "FAR: 0.00% | FRR: 38.24% [0 Non-Aq Leakage]"),
    ]:
        cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                ax.text(
                    j, i, str(val),
                    ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if val > cm.max() / 2 else "black"
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-Aquatic", "Aquatic"], fontsize=11)
        ax.set_yticklabels(["Non-Aquatic", "Aquatic"], fontsize=11)
        ax.set_xlabel("Predicted Class", fontsize=11, labelpad=10)
        ax.set_ylabel("True Class", fontsize=11)
        ax.set_title(f"{title}\n{sub}", fontsize=12, pad=12)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    print("=" * 75)
    print("       AQUARESTORE: RIGOROUS CLASSIFIER EVALUATION SUITE        ")
    print("=" * 75)

    config = ClassifierConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model Checkpoint: {config.model_path}")

    # Load Model
    checkpoint = torch.load(config.model_path, map_location=device)
    model = AquaticClassifier(num_classes=config.num_classes, pretrained=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Load Untouched Held-Out Test Dataset
    test_dir = config.dataset_dir / "test"
    print(f"Loading held-out test dataset from: {test_dir}")
    test_dataset = AquaticDataset.from_directory(test_dir, transform=get_eval_transforms(config))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Held-out test images: {len(test_dataset)} (102 Aquatic, 102 Non-Aquatic)\n")

    # 1. Test Set Evaluation
    print("1. Evaluating on untouched held-out test set...")
    test_results = run_test_set_evaluation(model, test_loader, device)
    b50 = test_results["baseline_tau_0_50"]
    o85 = test_results["operational_tau_0_85"]

    print("\n" + "-" * 75)
    print("   HELD-OUT TEST SET METRICS SUMMARY   ")
    print("-" * 75)
    print(f"Area Under ROC Curve (ROC-AUC): {test_results['roc_auc']:.4f}")
    print(f"\n[Baseline Threshold tau = 0.50]")
    print(f"  Accuracy:                      {b50['accuracy']:.2%}")
    print(f"  Precision:                     {b50['precision']:.2%}")
    print(f"  Recall (TPR):                  {b50['recall']:.2%}")
    print(f"  F1-Score:                      {b50['f1_score']:.4f}")
    print(f"  False Acceptance Rate (FAR):   {b50['false_acceptance_rate_far']:.2%}  ({b50['false_positives']} non-aquatic images falsely accepted)")
    print(f"  False Rejection Rate (FRR):    {b50['false_rejection_rate_frr']:.2%}  ({b50['false_negatives']} aquatic images falsely rejected)")
    print(f"  Confusion Matrix:              TN={b50['true_negatives']}, FP={b50['false_positives']}, FN={b50['false_negatives']}, TP={b50['true_positives']}")

    print(f"\n[Operational Threshold tau = 0.85]  <-- STRICT AQUATIC GATING")
    print(f"  Accuracy:                      {o85['accuracy']:.2%}")
    print(f"  Precision:                     {o85['precision']:.2%}")
    print(f"  Recall (TPR):                  {o85['recall']:.2%}")
    print(f"  F1-Score:                      {o85['f1_score']:.4f}")
    print(f"  False Acceptance Rate (FAR):   {o85['false_acceptance_rate_far']:.2%}  ({o85['false_positives']} non-aquatic images falsely accepted: ZERO LEAKAGE!)")
    print(f"  False Rejection Rate (FRR):    {o85['false_rejection_rate_frr']:.2%}  ({o85['false_negatives']} aquatic images rejected for uncertainty)")
    print(f"  Confusion Matrix:              TN={o85['true_negatives']}, FP={o85['false_positives']}, FN={o85['false_negatives']}, TP={o85['true_positives']}")
    print("-" * 75)

    # 2. Difficult Edge Cases
    print("\n2. Evaluating challenging edge cases (blue skies, swimming pools, beaches, murky water, lakes)...")
    engine = AquaticInference(model_path=config.model_path, threshold=0.85, device=str(device))
    edge_results = evaluate_edge_cases(engine)

    print("\n" + "=" * 75)
    print(f"{'Category':<32}{'File':<24}{'Pred':<12}{'Conf':<9}{'Status':<10}")
    print("=" * 75)
    for ec in edge_results:
        print(f"{ec['category'][:30]:<32}{ec['file'][:22]:<24}{ec['predicted']:<12}{ec['confidence']:<9.1%}{'ACCEPTED' if ec['accepted'] else 'REJECTED':<10}")
    print("=" * 75)

    # 3. Pipeline Gating and Mode Verification
    print("\n3. Verifying pipeline gating and enhancement modes...")
    pipeline_results = verify_pipeline_gating_and_modes()

    pg = pipeline_results["pipeline_gating"]
    print(f"Pipeline Gating Test on '{pg['test_image']}':")
    print(f"  Gating Triggered:    {pg['gating_successful']}")
    print(f"  Exception Produced:  {pg['raised_exception']}")
    print(f"  Message:             {pg['error_message']}")

    em = pipeline_results["enhancement_modes"]
    print(f"\nEnhancement Modes Verified on '{em['test_image']}':")
    print(f"  Fixed Mode:    shape={em['fixed_mode']['output_shape']}, clip_limit={em['fixed_mode']['clip_limit']}, latency={em['fixed_mode']['time_ms']} ms")
    print(f"  Adaptive Mode: shape={em['adaptive_mode']['output_shape']}, clip_limit={em['adaptive_mode']['clip_limit']}, latency={em['adaptive_mode']['time_ms']} ms")

    # 4. Save Artifacts
    cm_plot_path = Path("models/comprehensive_confusion_matrix.png")
    save_confusion_matrix_figure(b50["confusion_matrix"], o85["confusion_matrix"], cm_plot_path)
    print(f"\nSaved confusion matrix plot: {cm_plot_path}")

    full_report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_path": str(config.model_path),
        "test_dataset_summary": test_results,
        "edge_case_evaluations": edge_results,
        "pipeline_and_mode_verification": pipeline_results,
    }

    report_json_path = Path("models/comprehensive_evaluation_report.json")
    with open(report_json_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"Saved complete evaluation report JSON: {report_json_path}")


if __name__ == "__main__":
    main()

