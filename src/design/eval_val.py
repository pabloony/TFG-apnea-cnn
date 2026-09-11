# src/design/eval_val.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
)

from .config import IMPUT_DATA_DIR, EVAL_VAL_DIR, MODEL_BEST_PATH, VAL_METRICS_PATH, VAL_Y_PROB_PATH

DATA_DIR     = Path(IMPUT_DATA_DIR)
MODEL_PATH   = Path(MODEL_BEST_PATH)
X_VAL_PATH   = DATA_DIR / "X_val.npy"
Y_VAL_PATH   = DATA_DIR / "y_val.npy"
VAL_IDS_PATH = DATA_DIR / "val_ids.txt"


def _load_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _best_threshold_by_youden(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Maximiza (TPR - FPR). Devuelve (best_thr, best_score)."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    idx = int(np.argmax(youden))
    return float(thr[idx]), float(youden[idx])


def _best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Maximiza F1. Devuelve (best_thr, best_f1)."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)
    f1_thr = f1[1:]
    idx = int(np.argmax(f1_thr))
    return float(thr[idx]), float(f1_thr[idx])


def main() -> None:
    print("========== EVAL (VAL) ==========")
    print(f"[INFO] Model  : {MODEL_PATH}")
    print(f"[INFO] X_val  : {X_VAL_PATH}")
    print(f"[INFO] y_val  : {Y_VAL_PATH}")
    print(f"[INFO] out_dir: {EVAL_VAL_DIR}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not X_VAL_PATH.exists() or not Y_VAL_PATH.exists():
        raise FileNotFoundError("Missing X_val.npy or y_val.npy")

    # Cargar datos
    X_val = np.load(X_VAL_PATH, mmap_mode="r")
    y_val = np.load(Y_VAL_PATH, mmap_mode="r")

    # Asegurar dtypes
    if X_val.dtype != np.float32:
        print(f"[WARN] X_val dtype={X_val.dtype} -> casting to float32 (copy)")
        X_val = X_val.astype(np.float32, copy=False)

    y_val = (y_val > 0.5).astype(np.int32)

    print(f"[INFO] X_val shape={X_val.shape} dtype={X_val.dtype}")
    print(f"[INFO] y_val shape={y_val.shape} dtype={y_val.dtype}")
    print(f"[INFO] y_val counts: {dict(zip(*np.unique(y_val, return_counts=True)))}")

    # IDs (opcional)
    val_ids = _load_ids(VAL_IDS_PATH)
    if val_ids:
        print(f"[INFO] val_ids loaded: n={len(val_ids)} (show first 5): {val_ids[:5]}")

    # Cargar modelo
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Predicción
    print("[INFO] Predicting probabilities on VAL...")
    y_prob = model.predict(X_val, batch_size=256, verbose=1).reshape(-1).astype(np.float64)

    # Métricas threshold-free
    roc_auc = roc_auc_score(y_val, y_prob)
    pr_auc  = average_precision_score(y_val, y_prob)

    print("\n--- Threshold-free metrics ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")

    # Umbrales recomendados
    thr_youden, youden_score = _best_threshold_by_youden(y_val, y_prob)
    thr_f1, best_f1          = _best_threshold_by_f1(y_val, y_prob)

    print("\n--- Suggested thresholds ---")
    print(f"Best Youden threshold: {thr_youden:.4f} (TPR-FPR={youden_score:.4f})")
    print(f"Best F1 threshold    : {thr_f1:.4f} (F1={best_f1:.4f})")

    # Reporte a threshold=0.5 + thresholds óptimos
    def report_at_threshold(thr: float, name: str) -> None:
        y_pred = (y_prob >= thr).astype(np.int32)
        acc = accuracy_score(y_val, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp + 1e-12)

        print(f"\n--- Metrics @ {name} (thr={thr:.4f}) ---")
        print(f"Accuracy     : {acc:.4f}")
        print(f"Precision    : {prec:.4f}")
        print(f"Recall (Sens): {rec:.4f}")
        print(f"Specificity  : {spec:.4f}")
        print(f"F1           : {f1:.4f}")
        print("Confusion matrix [ [TN FP]\n                 [FN TP] ]:")
        print(cm)
        print("\nClassification report:")
        print(classification_report(y_val, y_pred, digits=4, zero_division=0))

    report_at_threshold(0.5, "default")
    report_at_threshold(thr_youden, "youden")
    report_at_threshold(thr_f1, "best_f1")

    # Guardar outputs
    out_dir = Path(EVAL_VAL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(Path(VAL_Y_PROB_PATH), y_prob.astype(np.float32))
    Path(VAL_METRICS_PATH).write_text(
        "\n".join(
            [
                f"ROC-AUC={roc_auc:.6f}",
                f"PR-AUC={pr_auc:.6f}",
                f"thr_youden={thr_youden:.6f}",
                f"thr_best_f1={thr_f1:.6f}",
                f"best_f1={best_f1:.6f}",
            ]
        )
        + "\n"
    )
    print(f"\n[OK] Saved val probabilities: {Path(VAL_Y_PROB_PATH)}")
    print(f"[OK] Saved val summary      : {Path(VAL_METRICS_PATH)}")
    print("[DONE]")


if __name__ == "__main__":
    main()