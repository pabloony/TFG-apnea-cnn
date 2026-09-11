#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_imput_trainval_full.py
----------------------------
Variante de main_imput_trainval_v2.py que NO hace el balanceo 1:1.

Diferencia con tu script actual:
  - Tu script actual: por cada sujeto, normaliza (z-score EDF/canal) y luego
    se queda solo con tantos negativos como positivos (np.random.choice).
  - Este script: normaliza igual, pero conserva TODOS los negativos del
    sujeto. El balanceo se hace después, en el entrenamiento, vía
    class_weight (Ronda 1) y/o filtrado de negativos "fáciles" (boosting,
    rondas siguientes) — no aquí.

Filtro de cohorte (NUEVO):
  - La lista de sujetos a usar se filtra según config.COHORTS_TO_USE
    (ej. ["STNF"] para usar solo STNF, ["STNF", "STLK"] para ambas).
  - Esto permite excluir STLK sin tocar este script: solo cambias
    COHORTS_TO_USE en config.py.

Reutiliza el MISMO split train/val que ya tienes, para que el experimento
sea comparable con tus fases anteriores:
  - Si existen train_ids.txt / val_ids.txt en IMPUT_RESULTS_DIR, los usa
    (tras aplicar el filtro de cohorte).
  - Si no existen, reproduce el split determinista (mismo SEED) que usa
    tu script actual, así que debería dar exactamente la misma partición.

NO modifica X_val.npy / y_val.npy existentes — la validación se queda
exactamente como está, para no romper comparabilidad de AUC entre fases.

Salida (en IMPUT_RESULTS_DIR):
  - X_train_full.npy, y_train_full.npy   (todos los negativos, normalizado)
  - train_ids_full.txt                    (mismos IDs que train_ids.txt,
                                            se guarda igual por trazabilidad)

USO (desde ProyectoPython3.0/, en Picasso3):
  python -m src.complementation.main_imput_trainval_full
"""

from pathlib import Path
import numpy as np

from .config import IMPUT_RESULTS_DIR, JOINED_RESULTS_DIR, COHORTS_TO_USE

# ── Parámetros (deben coincidir con los de tu script actual) ───────────
SEED = 42
VAL_FRACTION = 0.20
MIN_POS = 20            # mismo criterio de descarte de sujeto que ya usas
MIN_NEG_FOR_STATS = 10
EPS = 1e-6

np.random.seed(SEED)


def zscore_per_edf_per_channel(X: np.ndarray, y: np.ndarray,
                                eps: float = EPS,
                                min_neg: int = MIN_NEG_FOR_STATS) -> np.ndarray:
    """
    Idéntica a la de tu pipeline actual — no se toca.
    Normaliza por EDF y por canal (0=airflow, 1=spo2), usando μ/σ
    calculados SOLO con y==0 (clase normal) si hay suficientes negativos.
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


def load_normalize_full(npz_path: Path):
    """
    Carga joined_<ID>.npz y devuelve (X, y, (n_pos, n_neg)) con:
      - normalización z-score EDF/canal (igual que tu pipeline)
      - SIN balanceo 1:1 — se conservan todos los negativos
      - shape final (N, 2, 500, 1) float32

    Devuelve (None, None, None) si el sujeto no es válido, con el MISMO
    criterio de descarte que ya usas (MIN_POS), para no introducir sesgos
    nuevos al comparar contra tus experimentos anteriores.
    """
    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"    [ERROR] {npz_path.name} corrupto -> {type(e).__name__}: {e}")
        return None, None, None

    X = data["joined_windows"]  # (N, 2, 500)
    y = data["labels"]          # (N,)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    print(f"    [INFO] Segmentos totales: {len(y)} | Positivos: {n_pos} | Negativos: {n_neg}")

    if n_pos < MIN_POS:
        print(f"    [SKIP] Solo {n_pos} positivos (< {MIN_POS}). Sujeto descartado.")
        return None, None, None
    if n_pos == 0 or n_neg == 0:
        print("    [SKIP] Sin suficientes positivos/negativos. Sujeto descartado.")
        return None, None, None

    # Normalización z-score por EDF/canal (idéntica a tu pipeline actual)
    X = zscore_per_edf_per_channel(X, y)

    # Añadir eje para Conv2D: (N, 2, 500, 1)
    X = X[..., np.newaxis].astype(np.float32, copy=False)
    y = y.astype(np.int64, copy=False)

    # *** SIN BALANCEO 1:1 *** — se devuelven TODAS las ventanas del sujeto
    return X, y, (n_pos, n_neg)


def filter_by_cohort(files, cohorts):
    """
    Filtra una lista de Path joined_<ID>.npz quedándose solo con los IDs
    cuyo nombre contiene alguno de los prefijos/strings en `cohorts`.
    Si `cohorts` es None o vacío, no filtra (se devuelven todos).
    """
    if not cohorts:
        return files
    filtered = [f for f in files if any(c in f.stem for c in cohorts)]
    excluded = len(files) - len(filtered)
    print(f"[INFO] Filtro de cohorte activo: {cohorts} "
          f"-> {len(filtered)} sujetos incluidos, {excluded} excluidos.")
    return filtered


def get_train_val_split(joined_dir: Path, out_dir: Path):
    """
    Reutiliza train_ids.txt / val_ids.txt si existen (preferido).
    Si no existen, reproduce el split determinista de tu script actual.

    En ambos casos se aplica el filtro de cohorte definido en
    config.COHORTS_TO_USE antes de devolver la lista de ficheros.
    """
    train_ids_path = out_dir / "train_ids.txt"
    val_ids_path = out_dir / "val_ids.txt"

    files = sorted(joined_dir.glob("joined_*.npz"))
    if not files:
        raise FileNotFoundError(f"No hay joined_*.npz en {joined_dir.resolve()}")

    files = filter_by_cohort(files, COHORTS_TO_USE)
    if not files:
        raise RuntimeError(
            f"El filtro de cohorte {COHORTS_TO_USE} no ha dejado ningún "
            f"fichero. Revisa COHORTS_TO_USE en config.py."
        )

    def extract_id(p: Path) -> str:
        return p.stem.replace("joined_", "")

    by_id = {extract_id(p): p for p in files}

    if train_ids_path.exists() and val_ids_path.exists():
        print(f"[INFO] Reutilizando split existente: {train_ids_path.name} / {val_ids_path.name}")
        train_ids = train_ids_path.read_text(encoding="utf-8").split()
        train_files = [by_id[i] for i in train_ids if i in by_id]
        missing = [i for i in train_ids if i not in by_id]
        if missing:
            print(f"[INFO] {len(missing)} IDs de train_ids.txt no están en train_files "
                  f"(probablemente excluidos por COHORTS_TO_USE={COHORTS_TO_USE}, "
                  f"no es necesariamente un error).")
        return train_files

    print("[WARNING] No se encontraron train_ids.txt/val_ids.txt — "
          "reproduciendo split determinista (mismo SEED). Verifica que "
          "coincide con tu split real antes de usar estos resultados.")
    rng = np.random.RandomState(SEED)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    files_shuffled = [files[i] for i in idx]
    n_val = int(round(len(files_shuffled) * VAL_FRACTION))
    train_files = files_shuffled[n_val:]
    return train_files


def main():
    joined_dir = Path(JOINED_RESULTS_DIR)
    out_dir = Path(IMPUT_RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files = get_train_val_split(joined_dir, out_dir)
    print("-" * 60)
    print(f"[INFO] Cohortes incluidas (COHORTS_TO_USE): {COHORTS_TO_USE}")
    print(f"[INFO] Sujetos de TRAIN a procesar (full, sin balanceo): {len(train_files)}")
    print("-" * 60)

    X_list, y_list, ids_used = [], [], []
    total_pos, total_neg = 0, 0

    for i, f in enumerate(train_files, start=1):
        print(f"[INFO] ({i}/{len(train_files)}) Procesando: {f.name}")
        X, y, counts = load_normalize_full(f)
        if X is None:
            continue
        X_list.append(X)
        y_list.append(y)
        ids_used.append(f.stem.replace("joined_", ""))
        total_pos += counts[0]
        total_neg += counts[1]

    if not X_list:
        raise RuntimeError("Ningún sujeto válido. Revisa MIN_POS o las etiquetas.")

    X_train_full = np.concatenate(X_list, axis=0)
    y_train_full = np.concatenate(y_list, axis=0)

    # Shuffle global alineado (mismo criterio que tu pipeline: shuffle final)
    perm = np.random.permutation(len(y_train_full))
    X_train_full = X_train_full[perm]
    y_train_full = y_train_full[perm]

    print("-" * 60)
    print("[INFO] DATASET FULL (sin balanceo 1:1)")
    print("-" * 60)
    print(f"[INFO] Cohortes       : {COHORTS_TO_USE}")
    print(f"[INFO] Sujetos usados : {len(ids_used)}")
    print(f"[INFO] X_train_full   : {X_train_full.shape} | dtype={X_train_full.dtype}")
    print(f"[INFO] Positivos      : {total_pos}")
    print(f"[INFO] Negativos      : {total_neg}")
    if total_pos > 0:
        print(f"[INFO] Ratio neg:pos  : {total_neg / total_pos:.2f} : 1")
    print("-" * 60)

    np.save(out_dir / "X_train_full.npy", X_train_full)
    np.save(out_dir / "y_train_full.npy", y_train_full)
    (out_dir / "train_ids_full.txt").write_text("\n".join(ids_used) + "\n", encoding="utf-8")

    print("[OK] Guardado: X_train_full.npy, y_train_full.npy, train_ids_full.txt")
    print("[OK] X_val.npy / y_val.npy NO se han tocado.")


if __name__ == "__main__":
    main()