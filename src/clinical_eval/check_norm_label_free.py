#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_norm_label_free.py
------------------------
Compara la normalización z-score CON etiquetas (μ/σ solo de ventanas negativas,
como en zscore_per_edf_per_channel) frente a la versión LABEL-FREE (μ/σ de todas
las ventanas del sujeto). Cuantifica si usar etiquetas para elegir la población
de referencia introduce sesgo dependiente de la severidad (AHI).

No entrena ni infiere: solo carga joined_<ID>.npz y calcula estadísticos.
Submuestrea a 200k valores/canal (μ/σ estables) para ir rápido en el login node.
"""
import numpy as np, glob, os, re
import pandas as pd

# ============================================================
#  INTERRUPTOR DE COHORTE
#    "STLK" -> solo joined_STLK*.npz   (rápido; validación externa)
#    "STNF" -> solo joined_STNF*.npz
#    None   -> todos
# ============================================================
ONLY_COHORT = "STLK"

JOINED_DIR = "/mnt/home/users/ac_aux/portega/ProyectoPython3.0/data/complementation_results/joined_data"
GT_CSV     = "/mnt/home/users/ac_aux/portega/ProyectoPython3.0/ahi_labels.csv"
P_LOW, P_HIGH, MIN_NEG = 1, 99, 50
SUBSAMPLE  = 200_000
RNG        = np.random.default_rng(0)

def mu_sd(ref):
    lo, hi = np.percentile(ref, [P_LOW, P_HIGH])
    c = np.clip(ref, lo, hi)
    return c.mean(), c.std()

def maybe_subsample(a):
    return RNG.choice(a, SUBSAMPLE, replace=False) if a.size > SUBSAMPLE else a

# --- Ground truth: AHI a numérico de forma robusta ---
gt = pd.read_csv(GT_CSV, sep=";", decimal=",", dtype=str)     # todo como texto, sin sorpresas
gt["s_code"] = gt["s_code"].astype(str)
gt["ahi_num"] = pd.to_numeric(
    gt["ahi"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce"
)
# Mapa sid -> AHI (float). Si un sid está duplicado, toma el primero.
ahi_map = gt.dropna(subset=["ahi_num"]).drop_duplicates("s_code").set_index("s_code")["ahi_num"].to_dict()

def ahi_of(sid):
    v = ahi_map.get(sid, np.nan)
    try:
        return float(v)
    except Exception:
        return np.nan

# --- Selección de archivos según cohorte ---
if ONLY_COHORT:
    pattern = os.path.join(JOINED_DIR, f"joined_{ONLY_COHORT}*.npz")
else:
    pattern = os.path.join(JOINED_DIR, "joined_*.npz")
paths = sorted(glob.glob(pattern))
print(f"[INFO] cohorte={ONLY_COHORT or 'TODAS'} -> {len(paths)} sujetos", flush=True)

rows = []
for k, path in enumerate(paths, 1):
    sid = re.search(r"joined_(\w+)\.npz", os.path.basename(path)).group(1)
    print(f"  ({k}/{len(paths)}) {sid}", flush=True)
    try:
        d = np.load(path)
    except Exception as e:
        print(f"      [WARN] corrupto: {type(e).__name__}", flush=True)
        continue

    X, y = d["joined_windows"], d["labels"].astype(int)
    neg = (y == 0)
    use_neg = neg.sum() >= MIN_NEG
    pi  = 1 - neg.mean()
    ahi = ahi_of(sid)

    for ch in range(X.shape[1]):
        ref  = maybe_subsample((X[neg, ch, :] if use_neg else X[:, ch, :]).ravel())
        allx = maybe_subsample(X[:, ch, :].ravel())
        mu_n, sd_n = mu_sd(ref)
        mu_a, sd_a = mu_sd(allx)
        rows.append(dict(
            sid=sid, ch=ch, pi=float(pi), ahi=ahi,
            dmu_over_sd=float((mu_a - mu_n) / (sd_n + 1e-8)),
            sd_ratio=float(sd_a / (sd_n + 1e-8)),
        ))

df = pd.DataFrame(rows)
# Blindaje total: fuerza numérico en las columnas que se agregan
for col in ["pi", "ahi", "dmu_over_sd", "sd_ratio"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["cohorte"] = df.sid.str.extract(r"^(STNF|STLK)")

def safe_corr(a, b):
    m = a.notna() & b.notna()
    return a[m].corr(b[m]) if m.sum() > 2 else float("nan")

# ---- Resumen por canal ----
for ch, sub in df.groupby("ch"):
    nombre = "airflow" if ch == 0 else "spo2"
    corr = safe_corr(sub.dmu_over_sd.abs(), sub.ahi)
    n_ok = int((sub.ahi.notna()).sum())
    print(f"\n== canal {ch} ({nombre}) ==")
    print(f"  |Δμ/σ|   media={sub.dmu_over_sd.abs().mean():.4f}  max={sub.dmu_over_sd.abs().max():.4f}")
    print(f"  σall/σn  media={sub.sd_ratio.mean():.4f}  min={sub.sd_ratio.min():.4f}  max={sub.sd_ratio.max():.4f}")
    print(f"  corr(|Δμ/σ|, AHI) = {corr:.3f}  (n={n_ok})")

# ---- Top-5 mayor cambio de escala ----
print("\nTop-5 |σall/σn - 1| (cambio de escala más grande):")
df["esc_dev"] = (df.sd_ratio - 1).abs()
print(df.sort_values("esc_dev", ascending=False)
        .head(5)[["sid", "ch", "pi", "ahi", "dmu_over_sd", "sd_ratio"]]
        .to_string(index=False))

out = f"norm_label_free_deltas_{ONLY_COHORT or 'ALL'}.csv"
df.drop(columns="esc_dev").to_csv(out, index=False)
print(f"\n[OK] guardado {out}", flush=True)