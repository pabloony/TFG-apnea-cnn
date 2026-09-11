#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
diagnose_dataset_light.py
-------------------------
Versión ligera (anti-OOM) del diagnóstico.

- Muestrea K archivos joined_*.npz
- Para cada sujeto, subsamplea hasta N ventanas por clase (pos y neg)
- Calcula stats básicos por canal (mean/std/min/max/energy)
- Opcional: overfit-test con MLP mínimo sobre subset balanceado

También hace lo mismo para X_train.npy / y_train.npy si existen.

USO:
  python -m src.complementation.diagnose_dataset_light --sample_k 6 --per_class 800
  python -m src.complementation.diagnose_dataset_light --sample_k 6 --per_class 800 --overfit
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

JOINED_RESULTS_DIR = Path("data/complementation_results/joined_data/")
IMPUT_RESULTS_DIR  = Path("data/complementation_results/imput_data/")

XTRAIN_PATH = IMPUT_RESULTS_DIR / "X_train.npy"
YTRAIN_PATH = IMPUT_RESULTS_DIR / "y_train.npy"


def ensure_4d(X: np.ndarray) -> np.ndarray:
    if X.ndim == 3:
        return X[..., None]
    if X.ndim == 4:
        return X
    raise ValueError(f"Shape inesperada para X: {X.shape}")


def subsample_indices(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = y.astype(int)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return np.array([], dtype=int)

    npos = min(per_class, len(pos))
    nneg = min(per_class, len(neg))
    pos_sel = rng.choice(pos, size=npos, replace=False)
    neg_sel = rng.choice(neg, size=nneg, replace=False)

    idx = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(idx)
    return idx


def stats_basic(X: np.ndarray, y: np.ndarray) -> dict:
    """
    X: (N,2,500,1)
    y: (N,)
    """
    X = ensure_4d(X).astype(np.float64)
    y = y.astype(int)

    out = {}
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            out[cls] = None
            continue
        Xc = X[idx]  # (n,2,500,1)
        ch = {}
        for c, name in enumerate(("airflow", "spo2")):
            v = Xc[:, c, :, 0]
            ch[name] = {
                "n": int(v.shape[0]),
                "mean": float(v.mean()),
                "std": float(v.std()),
                "min": float(v.min()),
                "max": float(v.max()),
                "energy": float((v * v).mean()),
            }
        out[cls] = ch

    if out[0] is not None and out[1] is not None:
        out["deltas"] = {
            name: {
                "delta_mean": out[1][name]["mean"] - out[0][name]["mean"],
                "delta_std":  out[1][name]["std"]  - out[0][name]["std"],
                "ratio_energy": (out[1][name]["energy"] / out[0][name]["energy"]) if out[0][name]["energy"] != 0 else np.inf,
            }
            for name in ("airflow", "spo2")
        }
    return out


def print_stats(title: str, s: dict) -> None:
    print(f"\n=== {title} ===")
    for cls in (0, 1):
        if s.get(cls) is None:
            print(f"  Clase {cls}: (sin muestras)")
            continue
        print(f"  Clase {cls}:")
        for ch, d in s[cls].items():
            print(
                f"    {ch:7s} | n={d['n']:5d} mean={d['mean']:+.4f} std={d['std']:.4f} "
                f"min={d['min']:+.4f} max={d['max']:+.4f} energy={d['energy']:.4e}"
            )
    if "deltas" in s:
        print("  Deltas (pos - neg):")
        for ch, d in s["deltas"].items():
            print(
                f"    {ch:7s} | delta_mean={d['delta_mean']:+.4f} "
                f"delta_std={d['delta_std']:+.4f} ratio_energy={d['ratio_energy']:.4f}"
            )


def overfit_mlp(X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, lr: float, seed: int) -> None:
    import tensorflow as tf
    tf.random.set_seed(seed)
    X = ensure_4d(X).astype(np.float32)
    y = y.astype(np.float32)

    inp = tf.keras.layers.Input(shape=X.shape[1:])
    x = tf.keras.layers.Flatten()(inp)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.BinaryAccuracy(name="acc")],
    )
    print("\n[OVERFIT] MLP sanity…")
    hist = model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    print("[OVERFIT] last:", {k: v[-1] for k, v in hist.history.items()})


def diagnose_joined(sample_k: int, per_class: int, seed: int, overfit: bool,
                    overfit_n: int, overfit_epochs: int, batch_size: int, lr: float) -> None:
    files = sorted(JOINED_RESULTS_DIR.glob("joined_STNF*.npz"))
    if not files:
        raise FileNotFoundError(f"No encuentro joined_*.npz en {JOINED_RESULTS_DIR.resolve()}")

    rng = np.random.default_rng(seed)
    k = min(sample_k, len(files))
    chosen = sorted(rng.choice(files, size=k, replace=False))

    print(f"[INFO] JOINED dir: {JOINED_RESULTS_DIR.resolve()}")
    print(f"[INFO] Total joined files: {len(files)} | Muestreados: {k} | per_class={per_class}")

    for i, p in enumerate(chosen, start=1):
        with np.load(p, allow_pickle=False) as d:
            X = d["joined_windows"]
            y = d["labels"].astype(int)

        print(f"\n--- ({i}/{k}) {p.name} ---")
        print("X shape:", X.shape, "dtype:", X.dtype, "| y:", y.shape, y.dtype)
        uniq, cnt = np.unique(y, return_counts=True)
        print("y counts:", dict(zip(uniq.tolist(), cnt.tolist())))

        # Subsample para stats
        idx = subsample_indices(y, per_class=per_class, seed=seed)
        if idx.size == 0:
            print("[WARN] no hay ambas clases, salto.")
            continue

        Xs = ensure_4d(X[idx])
        ys = y[idx]

        s = stats_basic(Xs, ys)
        print_stats(f"Stats (subsample {len(ys)} ventanas)", s)

        if overfit:
            # subset balanceado para overfit
            Xb, yb = Xs, ys
            # reducimos a overfit_n balanceado exacto
            rng2 = np.random.default_rng(seed)
            pos = np.where(yb == 1)[0]
            neg = np.where(yb == 0)[0]
            n_each = min(len(pos), len(neg), overfit_n // 2)
            sel = np.concatenate([rng2.choice(pos, n_each, replace=False), rng2.choice(neg, n_each, replace=False)])
            rng2.shuffle(sel)
            overfit_mlp(Xb[sel], yb[sel], epochs=overfit_epochs, batch_size=batch_size, lr=lr, seed=seed)


def diagnose_imput(per_class: int, seed: int, overfit: bool,
                   overfit_n: int, overfit_epochs: int, batch_size: int, lr: float) -> None:
    print(f"\n[INFO] IMPUT dir: {IMPUT_RESULTS_DIR.resolve()}")

    if not XTRAIN_PATH.exists() or not YTRAIN_PATH.exists():
        print("[WARN] No existen X_train.npy / y_train.npy. Me salto esta parte.")
        return

    X = np.load(XTRAIN_PATH, mmap_mode="r")  # mmap para no petar RAM
    y = np.load(YTRAIN_PATH).astype(int)

    print("[INFO] X_train:", X.shape, X.dtype, "| y_train:", y.shape, y.dtype)
    uniq, cnt = np.unique(y, return_counts=True)
    print("[INFO] y counts:", dict(zip(uniq.tolist(), cnt.tolist())))

    idx = subsample_indices(y, per_class=per_class, seed=seed)
    if idx.size == 0:
        print("[WARN] X_train no tiene ambas clases?")
        return

    Xs = ensure_4d(np.asarray(X[idx]))  # convertimos solo el trozo
    ys = y[idx]

    s = stats_basic(Xs, ys)
    print_stats(f"Stats global X_train (subsample {len(ys)} ventanas)", s)

    if overfit:
        rng = np.random.default_rng(seed)
        pos = np.where(ys == 1)[0]
        neg = np.where(ys == 0)[0]
        n_each = min(len(pos), len(neg), overfit_n // 2)
        sel = np.concatenate([rng.choice(pos, n_each, replace=False), rng.choice(neg, n_each, replace=False)])
        rng.shuffle(sel)
        overfit_mlp(Xs[sel], ys[sel], epochs=overfit_epochs, batch_size=batch_size, lr=lr, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_k", type=int, default=6, help="Nº de sujetos joined a muestrear.")
    ap.add_argument("--per_class", type=int, default=800, help="Máx ventanas por clase (pos/neg) para stats.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--overfit_n", type=int, default=512)
    ap.add_argument("--overfit_epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    if not JOINED_RESULTS_DIR.exists():
        raise FileNotFoundError(f"JOINED_RESULTS_DIR no existe: {JOINED_RESULTS_DIR.resolve()}")

    diagnose_joined(
        sample_k=args.sample_k,
        per_class=args.per_class,
        seed=args.seed,
        overfit=args.overfit,
        overfit_n=args.overfit_n,
        overfit_epochs=args.overfit_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    diagnose_imput(
        per_class=args.per_class,
        seed=args.seed,
        overfit=args.overfit,
        overfit_n=args.overfit_n,
        overfit_epochs=args.overfit_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    print("\n[OK] Diagnóstico light completado.")


if __name__ == "__main__":
    main()
