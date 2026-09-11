#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_imput_trainval_v2.py
--------------------------
Construye X_train/y_train y X_val/y_val a partir de joined_*.npz,
haciendo split SUBJECT-WISE (por archivo), balanceo 1:1 y normalización
z-score por EDF y por canal (usando solo clase 0 si hay suficientes).

IMPORTANTE: Los sujetos STLK van SIEMPRE a training.
            El val set se construye exclusivamente con sujetos STNF
            para mantener comparabilidad con experimentos anteriores.

Salida (en IMPUT_RESULTS_DIR):
  - X_train.npy, y_train.npy
  - X_val.npy,   y_val.npy
  - train_ids.txt, val_ids.txt
"""

from pathlib import Path
import numpy as np
from .config import IMPUT_RESULTS_DIR, JOINED_RESULTS_DIR, COHORTS_TO_USE

# ---------------------------
# Parámetros
# ---------------------------
SEED         = 42
VAL_FRACTION = 0.20
MIN_POS      = 20    # sujetos con < MIN_POS positivos → descartados
MIN_NEG      = 10    # mínimos negativos para calcular μ/σ con clase 0
EPS          = 1e-6

np.random.seed(SEED)


# ---------------------------
# Normalización z-score por EDF y por canal
# ---------------------------
def zscore_per_edf_per_channel(X: np.ndarray, y: np.ndarray,
                                eps: float = EPS,
                                min_neg: int = MIN_NEG,
                                p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """
    X : (N, 2, 500) float
    y : (N,) int
    Normaliza cada canal (0=airflow, 1=spo2) por separado.
    Usa μ/σ calculados SOLO con y==0 (clase normal) si hay suficientes,
    RECORTANDO esa señal de referencia al percentil [p_low, p_high]
    antes de calcular μ/σ (Nassi et al.) -- evita que outliers de sensor
    inflen σ y aplasten el rango dinámico de la señal normal.
    La señal completa (sin recortar) se normaliza igual con esa μ/σ.
    Si no hay suficientes negativos, usa todas las ventanas como fallback.
    Devuelve X normalizado como float32.
    """
    Xn = X.astype(np.float32, copy=False)

    neg_mask = (y == 0)
    use_neg  = int(np.sum(neg_mask)) >= min_neg

    for ch in range(Xn.shape[1]):  # 0=airflow, 1=spo2
        ref = Xn[neg_mask, ch, :] if use_neg else Xn[:, ch, :]
        ref_flat = ref.ravel()

        lo, hi = np.percentile(ref_flat, [p_low, p_high])
        clipped = np.clip(ref_flat, lo, hi)

        mu = clipped.mean(dtype=np.float32)
        sd = clipped.std(dtype=np.float32)

        if sd < eps:
            sd = 1.0  # evita dividir por ~0 si el canal es casi constante

        Xn[:, ch, :] = (Xn[:, ch, :] - mu) / (sd + eps)

    return Xn

# ---------------------------
# Carga + normalización + balanceo (por sujeto)
# ---------------------------
def load_normalize_balance(npz_path: Path):
    """
    Carga joined_<ID>.npz y devuelve (X_bal, y_bal, (n_pos, n_neg)):
      - normalizado z-score por EDF y canal
      - balanceado 1:1 (mismos positivos que negativos)
      - shape final (N, 2, 500, 1) float32

    Devuelve (None, None, None) si el sujeto no es válido.
    """
    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"    [ERROR] {npz_path.name} corrupto → {type(e).__name__}: {e}")
        return None, None, None

    X = data["joined_windows"]  # (N, 2, 500)
    y = data["labels"]          # (N,)

    N_total = len(y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos   = len(pos_idx)
    n_neg   = len(neg_idx)

    print(f"    [INFO] Segmentos totales: {N_total}")
    print(f"    [INFO] Positivos: {n_pos} | Negativos: {n_neg}")

    if n_pos < MIN_POS:
        print(f"    [SKIP] Solo {n_pos} positivos (< {MIN_POS}). Sujeto descartado.")
        return None, None, None

    if n_pos == 0 or n_neg == 0:
        print(f"    [SKIP] Sin suficientes positivos/negativos. Sujeto descartado.")
        return None, None, None

    if n_neg < n_pos:
        print(f"    [SKIP] Menos negativos ({n_neg}) que positivos ({n_pos}). Sujeto descartado.")
        return None, None, None

    # Normalización z-score por EDF/canal (antes del balanceo)
    X = zscore_per_edf_per_channel(X, y)

    # Añadir eje para Conv2D: (N, 2, 500, 1)
    X = X[..., np.newaxis].astype(np.float32, copy=False)

    # Balanceo 1:1
    chosen_neg   = np.random.choice(neg_idx, size=n_pos, replace=False)
    balanced_idx = np.concatenate([pos_idx, chosen_neg])

    Xb = X[balanced_idx]
    yb = y[balanced_idx].astype(np.int64, copy=False)

    # Shuffle alineado
    perm = np.random.permutation(len(yb))
    Xb = Xb[perm]
    yb = yb[perm]

    print(f"    [OK] z-score + balanceado: {len(yb)} ventanas (pos={n_pos}, neg={len(chosen_neg)})")

    return Xb, yb, (n_pos, n_neg)


# ---------------------------
# Utilidad: guardar lista de IDs
# ---------------------------
def save_ids(path: Path, ids: list):
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


# ---------------------------
# Main
# ---------------------------
def main():
    joined_dir = Path(JOINED_RESULTS_DIR)
    out_dir    = Path(IMPUT_RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Buscar sujetos
    files = sorted(joined_dir.glob("joined_*.npz"))
    if COHORTS_TO_USE:
        files = [f for f in files if any(c in f.stem for c in COHORTS_TO_USE)]
        print(f"[INFO] Filtro cohorte {COHORTS_TO_USE}: {len(files)} sujetos incluidos")
    if not files:
        raise FileNotFoundError(f"No hay joined_*.npz en {joined_dir.resolve()}")

    def extract_id(p: Path) -> str:
        return p.stem.replace("joined_", "")  # joined_STNF00042.npz → STNF00042

    # ── Split: STLK siempre a train, val solo de STNF ────────────────
    stlk_files = [f for f in files if "STLK" in f.stem]
    stnf_files = [f for f in files if "STLK" not in f.stem]

    # Shuffle determinista solo sobre STNF
    rng = np.random.RandomState(SEED)
    idx = np.arange(len(stnf_files))
    rng.shuffle(idx)
    stnf_files = [stnf_files[i] for i in idx]

    n_val       = int(round(len(stnf_files) * VAL_FRACTION))
    val_files   = stnf_files[:n_val]
    train_files = stnf_files[n_val:] + stlk_files  # STNF train + todos los STLK

    train_ids = [extract_id(p) for p in train_files]
    val_ids   = [extract_id(p) for p in val_files]

    print("-" * 60)
    print(f"[INFO] JOINED dir   : {joined_dir.resolve()}")
    print(f"[INFO] Output dir   : {out_dir.resolve()}")
    print(f"[INFO] Total sujetos: {len(files)}")
    print(f"[INFO] STNF         : {len(stnf_files)} | STLK: {len(stlk_files)}")
    print(f"[INFO] Train        : {len(train_files)} (cohortes: {COHORTS_TO_USE})")
    print(f"[INFO] Val          : {len(val_files)} (solo STNF)")
    print(f"[INFO] SEED={SEED} | VAL_FRACTION={VAL_FRACTION}")
    print("-" * 60)

    # ── Procesar splits ───────────────────────────────────────────────
    def process_split(split_name: str, split_files: list) -> tuple:
        X_list, y_list = [], []
        used = 0
        for i, f in enumerate(split_files, start=1):
            subj_id = extract_id(f)
            print("-" * 60)
            print(f"[INFO] ({i}/{len(split_files)}) [{split_name}] {subj_id}")

            Xb, yb, counts = load_normalize_balance(f)
            if Xb is None:
                continue

            used += 1
            X_list.append(Xb)
            y_list.append(yb)

        return X_list, y_list, used

    print("\n[INFO] ── Procesando TRAIN ──")
    Xtr_list, ytr_list, used_train = process_split("TRAIN", train_files)

    print("\n[INFO] ── Procesando VAL ──")
    Xva_list, yva_list, used_val = process_split("VAL", val_files)

    # ── Validación del split ──────────────────────────────────────────
    if used_train == 0 or used_val == 0:
        raise RuntimeError(
            f"Split inválido: used_train={used_train}, used_val={used_val}. "
            f"Reduce MIN_POS o revisa labels en joined."
        )

    # ── Concatenar ────────────────────────────────────────────────────
    X_train = np.concatenate(Xtr_list, axis=0).astype(np.float32, copy=False)
    y_train = np.concatenate(ytr_list, axis=0).astype(np.int64,   copy=False)
    X_val   = np.concatenate(Xva_list, axis=0).astype(np.float32, copy=False)
    y_val   = np.concatenate(yva_list, axis=0).astype(np.int64,   copy=False)

    # ── Resumen final ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[INFO] DATASETS FINALES")
    print("=" * 60)
    print(f"[INFO] Train → sujetos: {used_train} | shape: {X_train.shape} | dtype: {X_train.dtype}")
    print(f"[INFO]          y: {y_train.shape} | 0={np.sum(y_train==0)} 1={np.sum(y_train==1)}")
    print(f"[INFO] Val   → sujetos: {used_val}   | shape: {X_val.shape}   | dtype: {X_val.dtype}")
    print(f"[INFO]          y: {y_val.shape}   | 0={np.sum(y_val==0)} 1={np.sum(y_val==1)}")
    print("=" * 60)

    # ── Guardar ───────────────────────────────────────────────────────
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy",   X_val)
    np.save(out_dir / "y_val.npy",   y_val)

    save_ids(out_dir / "train_ids.txt", train_ids)
    save_ids(out_dir / "val_ids.txt",   val_ids)

    print(f"\n[OK] Guardado en: {out_dir.resolve()}")
    print("[OK] train_ids.txt y val_ids.txt guardados.")


if __name__ == "__main__":
    main()