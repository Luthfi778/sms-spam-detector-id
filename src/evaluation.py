"""Perhitungan metrik evaluasi (dipakai oleh train.py dan skrip IndoBERT)."""
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, roc_auc_score,
)

from .config import LABELS, POS_LABEL


def metrics_block(y_true, y_pred, proba_spam) -> dict:
    rep = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    pos = rep[POS_LABEL]
    is_spam = (y_true == POS_LABEL)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_spam": float(pos["precision"]),
        "recall_spam": float(pos["recall"]),
        "f1_spam": float(pos["f1-score"]),
        "precision_macro": float(rep["macro avg"]["precision"]),
        "recall_macro": float(rep["macro avg"]["recall"]),
        "f1_macro": float(rep["macro avg"]["f1-score"]),
        "roc_auc": float(roc_auc_score(is_spam, proba_spam)),
        "pr_auc": float(average_precision_score(is_spam, proba_spam)),
        "per_class": {
            lab: {
                "precision": float(rep[lab]["precision"]),
                "recall": float(rep[lab]["recall"]),
                "f1": float(rep[lab]["f1-score"]),
                "support": int(rep[lab]["support"]),
            }
            for lab in LABELS
        },
        # baris = label asli, kolom = prediksi, urutan LABELS: [[TN, FP], [FN, TP]]
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }
