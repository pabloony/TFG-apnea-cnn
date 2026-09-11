#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_imput_trainval.py
----------------------
Construye X_train/y_train y X_val/y_val a partir de joined_*.npz,
haciendo split SUBJECT-WISE (por archivo), balanceo 1:1 y normalización
z-score por EDF y por canal (usando solo clase 0 si hay suficientes).

Salida (en IMPUT_RESULTS_DIR):
  - X_train.npy, y_train.npy
  - X_val.npy,   y_val.npy
  - train_ids.txt, val_ids.txt   (opcional, útil para trazabilidad)
"""

from pathlib import Path
import numpy as np
from .config import IMPUT_RESULTS_DIR, JOINED_RESULTS_DIR

# ---------------------------
# Parámetros
# ---------------------------
SEED = 42
VAL_FRACTION = 0.20
MIN_POS = 20          # si un sujeto tiene < MIN_POS positivos → se descarta
MIN_NEG_FOR_STATS = 10  # mínimos negativos para calcular μ/σ con clase 0
EPS = 1e-6

np.random.seed(SEED)


def zscore_per_edf_per_channel(X: np.ndarray, y: np.ndarray,
                               eps: float = EPS,
                               min_neg: int = MIN_NEG_FOR_STATS) -> np.ndarray:
    """
    X: (N, 2, 500) float
    y: (N,) int
    Devuelve X normalizado (float32) por canal, stats por EDF:
      - μ/σ calculados con y==0 si hay suficientes
      - fallback a todas las ventanas si no hay suficientes y==0
    """
    Xn = X.astype(np.float32, copy=False)

    neg_mask = (y == 0)
    use_neg = int(np.sum(neg_mask)) >= min_neg

    for ch in range(Xn.shape[1]):  # 0=airflow, 1=spo2
        ref = Xn[neg_mask, ch, :] if use_neg else Xn[:, ch, :]
        mu = ref.mean(dtype=np.float32)
        sd = ref.std(dtype=np.float32)

        if sd < eps:
            sd = 1.0

        Xn[:, ch, :] = (Xn[:, ch, :] - mu) / (sd + eps)

    return Xn


def load_normalize_balance(npz_path: Path):
    """
    Carga joined_<ID>.npz y devuelve (X_bal, y_bal) ya:
      - normalizado z-score EDF/canal
      - balanceado 1:1
      - con shape final (N, 2, 500, 1) float32
    Si el sujeto no es válido, devuelve (None, None, None).
    """
    data = np.load(npz_path)
    X = data["joined_windows"]   # (N, 2, 500)
    y = data["labels"]           # (N,)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos < MIN_POS or n_pos == 0 or n_neg == 0:
        return None, None, None

    # Normalización por EDF/canal (antes del balanceo)
    X = zscore_per_edf_per_channel(X, y)

    # Balanceo 1:1 (mismos negativos que positivos)
    if n_neg < n_pos:
        return None, None, None

    chosen_neg = np.random.choice(neg_idx, size=n_pos, replace=False)
    balanced_idx = np.concatenate([pos_idx, chosen_neg])

    Xb = X[balanced_idx]                       # (2*n_pos, 2, 500)
    yb = y[balanced_idx].astype(np.int64, copy=False)

    # Añadir eje final para Conv2D
    Xb = Xb[..., np.newaxis].astype(np.float32, copy=False)  # (2*n_pos, 2, 500, 1)

    # Shuffle alineado
    perm = np.random.permutation(len(yb))
    Xb = Xb[perm]
    yb = yb[perm]

    return Xb, yb, (n_pos, n_neg)


def save_ids(path: Path, ids: list[str]):
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def main():
    joined_dir = Path(JOINED_RESULTS_DIR)
    out_dir = Path(IMPUT_RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(joined_dir.glob("joined_*.npz"))
    if not files:
        raise FileNotFoundError(f"No hay joined_*.npz en {joined_dir.resolve()}")

    # Shuffle determinista de sujetos para split
    rng = np.random.RandomState(SEED)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    files = [files[i] for i in idx]

    n_val = int(round(len(files) * VAL_FRACTION))
    val_files = files[:n_val]
    train_files = files[n_val:]

    def extract_id(p: Path) -> str:
        # joined_STNF00042.npz -> STNF00042
        return p.stem.replace("joined_", "")

    train_ids = [extract_id(p) for p in train_files]
    val_ids = [extract_id(p) for p in val_files]

    print("-" * 60)
    print(f"[INFO] JOINED dir: {joined_dir.resolve()}")
    print(f"[INFO] Total sujetos: {len(files)} | Train: {len(train_files)} | Val: {len(val_files)}")
    print(f"[INFO] SEED={SEED} | VAL_FRACTION={VAL_FRACTION}")
    print("-" * 60)

    Xtr_list, ytr_list = [], []
    Xva_list, yva_list = [], []
    used_train, used_val = 0, 0

    def process_split(split_name: str, split_files: list[Path], X_list, y_list) -> int:
        used = 0
        for i, f in enumerate(split_files, start=1):
            subj_id = extract_id(f)
            Xb, yb, counts = load_normalize_balance(f)

            if Xb is None:
                print(f"[SKIP] {split_name} {subj_id} (no cumple MIN_POS/balanceo)")
                continue

            used += 1
            X_list.append(Xb)
            y_list.append(yb)

            n_pos, n_neg = counts
            print(f"[OK] {split_name} {subj_id} | pos={n_pos} neg={n_neg} -> balanced={len(yb)}")
        return used

    print("\n[INFO] Procesando TRAIN...")
    used_train = process_split("TRAIN", train_files, Xtr_list, ytr_list)

    print("\n[INFO] Procesando VAL...")
    used_val = process_split("VAL", val_files, Xva_list, yva_list)

    if used_train == 0 or used_val == 0:
        raise RuntimeError(f"Split inválido: used_train={used_train}, used_val={used_val}. "
                           f"Reduce MIN_POS o revisa labels en joined.")

    X_train = np.concatenate(Xtr_list, axis=0).astype(np.float32, copy=False)
    y_train = np.concatenate(ytr_list, axis=0).astype(np.int64, copy=False)

    X_val = np.concatenate(Xva_list, axis=0).astype(np.float32, copy=False)
    y_val = np.concatenate(yva_list, axis=0).astype(np.int64, copy=False)

    print("\n" + "-" * 60)
    print("[INFO] DATASETS FINALES")
    print("-" * 60)
    print(f"[INFO] Train sujetos usados: {used_train} | X_train: {X_train.shape} {X_train.dtype} | y: {y_train.shape}")
    print(f"[INFO] Val   sujetos usados: {used_val}   | X_val  : {X_val.shape} {X_val.dtype}   | y: {y_val.shape}")
    print(f"[INFO] Train counts: 0={np.sum(y_train==0)} 1={np.sum(y_train==1)}")
    print(f"[INFO] Val   counts: 0={np.sum(y_val==0)} 1={np.sum(y_val==1)}")
    print("-" * 60)

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "y_val.npy", y_val)

    # Guardar IDs para trazabilidad
    save_ids(out_dir / "train_ids.txt", train_ids)
    save_ids(out_dir / "val_ids.txt", val_ids)

    print(f"[OK] Guardado en: {out_dir.resolve()}")
    print("[OK] train_ids.txt y val_ids.txt guardados.")


if __name__ == "__main__":
    main()
