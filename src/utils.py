import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    genuine_scores = y_scores[y_true == 0]
    spoof_scores = y_scores[y_true == 1]

    if len(genuine_scores) == 0 or len(spoof_scores) == 0:
        return 1.0

    thresholds = np.linspace(0, 1, 1000)
    far_list = np.zeros(len(thresholds))
    frr_list = np.zeros(len(thresholds))

    for i, thresh in enumerate(thresholds):
        far_list[i] = np.sum(genuine_scores >= thresh) / len(genuine_scores)
        frr_list[i] = np.sum(spoof_scores < thresh) / len(spoof_scores)

    diff = np.abs(far_list - frr_list)
    eer = (far_list[diff.argmin()] + frr_list[diff.argmin()]) / 2
    return float(eer)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray
) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    eer = compute_eer(y_true, y_scores)

    tn, fp, fn, tp = cm.ravel()
    genuine_acc = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    deepfake_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        'accuracy': acc * 100,
        'eer': eer * 100,
        'f1_score': f1 * 100,
        'genuine_accuracy': genuine_acc * 100,
        'deepfake_accuracy': deepfake_acc * 100,
        'confusion_matrix': cm,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def print_metrics_report(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy:    {metrics['accuracy']:.2f}%  {'PASS' if metrics['accuracy'] >= 80 else 'FAIL'}")
    print(f"Equal Error Rate:    {metrics['eer']:.2f}%  {'PASS' if metrics['eer'] <= 12 else 'FAIL'}")
    print(f"F1 Score:            {metrics['f1_score']:.2f}%  {'PASS' if metrics['f1_score'] >= 80 else 'FAIL'}")
    print(f"Genuine Accuracy:    {metrics['genuine_accuracy']:.2f}%  {'PASS' if metrics['genuine_accuracy'] >= 75 else 'FAIL'}")
    print(f"Deepfake Accuracy:   {metrics['deepfake_accuracy']:.2f}%  {'PASS' if metrics['deepfake_accuracy'] >= 75 else 'FAIL'}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={metrics['tn']:>6}  FP={metrics['fp']:>6}")
    print(f"  FN={metrics['fn']:>6}  TP={metrics['tp']:>6}")
    print("=" * 60)

    all_pass = (
        metrics['accuracy'] >= 80 and
        metrics['eer'] <= 12 and
        metrics['f1_score'] >= 80 and
        metrics['genuine_accuracy'] >= 75 and
        metrics['deepfake_accuracy'] >= 75
    )
    print(f"\nVERIFICATION: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return all_pass


def plot_confusion_matrix(cm: np.ndarray, save_path: str = None) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Genuine', 'Deepfake'],
                yticklabels=['Genuine', 'Deepfake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_eer_curve(y_true: np.ndarray, y_scores: np.ndarray, save_path: str = None) -> None:
    genuine_scores = y_scores[y_true == 0]
    spoof_scores = y_scores[y_true == 1]

    thresholds = np.linspace(0, 1, 1000)
    far = np.zeros(len(thresholds))
    frr = np.zeros(len(thresholds))

    for i, t in enumerate(thresholds):
        far[i] = np.sum(genuine_scores >= t) / len(genuine_scores)
        frr[i] = np.sum(spoof_scores < t) / len(spoof_scores)

    diff = np.abs(far - frr)
    eer_idx = diff.argmin()
    eer_val = (far[eer_idx] + frr[eer_idx]) / 2

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, far, 'b-', label='FAR (False Acceptance Rate)', linewidth=2)
    plt.plot(thresholds, frr, 'r-', label='FRR (False Rejection Rate)', linewidth=2)
    plt.plot(thresholds[eer_idx], eer_val, 'go', markersize=10, label=f'EER = {eer_val:.4f}')
    plt.axhline(y=eer_val, color='green', linestyle='--', alpha=0.5)
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate')
    plt.title('EER Curve — FAR vs FRR')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_score_distribution(y_true: np.ndarray, y_scores: np.ndarray, save_path: str = None) -> None:
    genuine = y_scores[y_true == 0]
    spoof = y_scores[y_true == 1]

    plt.figure(figsize=(8, 5))
    plt.hist(genuine, bins=50, alpha=0.6, label='Genuine (Human)', color='green', density=True)
    plt.hist(spoof, bins=50, alpha=0.6, label='Deepfake (AI)', color='red', density=True)
    plt.xlabel('Model Score (confidence for Deepfake)')
    plt.ylabel('Density')
    plt.title('Score Distribution — Genuine vs Deepfake')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
