import numpy as np
from pathlib import Path
from .config import IMPUT_RESULTS_DIR, JOINED_RESULTS_DIR

MIN_POS = 20   # si un sujeto tiene menos de X positivos → se descarta
SEED = 42      # para reproducibilidad
np.random.seed(SEED)

# 1. Listar sujetos
IMPUT_RESULTS_DIR  = Path(IMPUT_RESULTS_DIR)
JOINED_RESULTS_DIR = Path(JOINED_RESULTS_DIR)
files = sorted(JOINED_RESULTS_DIR.glob("joined_*.npz"))

train_files = files   # de momento todo a training

print("-"*60)
print(f"[INFO] Sujetos encontrados: {len(train_files)}")
print("-"*60)


def zscore_per_edf_per_channel(X, y, eps=1e-6, min_neg=10):
    """
    Normaliza X por EDF (este npz) y por canal (0=airflow, 1=spo2),
    usando μ/σ calculados SOLO con y==0 (clase normal) si hay suficientes.
    Si no hay suficientes negativos, usa todas las ventanas como fallback.

    X: (N, 2, 500) float
    y: (N,) int
    """
    Xn = X.astype(np.float32, copy=False)

    neg_mask = (y == 0)
    use_neg = int(np.sum(neg_mask)) >= min_neg

    for ch in range(Xn.shape[1]):  # 0 y 1
        if use_neg:
            ref = Xn[neg_mask, ch, :]
        else:
            ref = Xn[:, ch, :]

        mu = ref.mean(dtype=np.float32)
        sd = ref.std(dtype=np.float32)

        if sd < eps:
            # Si el canal fuese casi constante (raro), evita dividir por ~0
            sd = 1.0

        Xn[:, ch, :] = (Xn[:, ch, :] - mu) / (sd + eps)

    return Xn


# Función de carga + normalización + balanceo
def load_and_balance(npz_path):
    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"    [ERROR] {npz_path.name} corrupto -> {type(e).__name__}")
        return None, None #333 cazamos errores de posibles archivos corruptos sin parar todo el proceso

    X = data["joined_windows"]     # (N, 2, 500)
    y = data["labels"]             # (N,)

    N_total = len(y)

    # Índices
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    print(f"    [INFO] Segmentos totales: {N_total}")
    print(f"    [INFO] Positivos: {n_pos}, Negativos: {n_neg}")

    # Opción: descartar sujetos con pocos positivos
    if n_pos < MIN_POS:
        print(f"    [WARNING] Solo {n_pos} positivos (< {MIN_POS}). Se descarta sujeto.")
        return None, None

    # Sin positivos o sin negativos → no se puede balancear
    if n_pos == 0 or n_neg == 0:
        print(f"    [WARNING] Sin suficientes positivos/negativos. Sujeto descartado.")
        return None, None

    # --- NORMALIZACIÓN POR EDF (antes del balanceo) ---
    # Nota: asumimos que tus ventanas ya son válidas/filtradas como dices.
    X = zscore_per_edf_per_channel(X, y)

    # Añadir eje "canal de color" para Conv2D: (N, 2, 500, 1)
    X = X[..., np.newaxis].astype(np.float32, copy=False)

    # Balanceo 1:1 → elegir aleatoriamente tantos negativos como positivos
    if n_neg >= n_pos:
        chosen_neg = np.random.choice(neg_idx, size=n_pos, replace=False)
    else:
        print(f"    [WARNING] Menos negativos que positivos; se toman todos los negativos.")
        return None, None

    balanced_idx = np.concatenate([pos_idx, chosen_neg])

    X_bal = X[balanced_idx]
    y_bal = y[balanced_idx].astype(np.int64, copy=False)

    # SHUFFLE (alineado)
    perm = np.random.permutation(len(y_bal))
    X_bal = X_bal[perm]
    y_bal = y_bal[perm]

    print(f"    [OK] Normalizado(z-score EDF/canal) + balanceado: {len(y_bal)} (Pos={n_pos}, Neg={len(chosen_neg)})")

    return X_bal, y_bal


# 2. Procesar todos los sujetos
X_train_list = []
y_train_list = []

subjects_aceptados = 0
total_pos = 0
total_neg = 0

for i, f in enumerate(train_files, start=1):
    print("-"*60)
    print(f"[INFO] ({i}/{len(train_files)}) Procesando: {f.name}")

    Xb, yb = load_and_balance(f)
    if Xb is None:
        continue

    subjects_aceptados += 1
    X_train_list.append(Xb)
    y_train_list.append(yb)

    total_pos += np.sum(yb == 1)
    total_neg += np.sum(yb == 0)


# 3. Construir dataset final
if subjects_aceptados == 0:
    raise RuntimeError("Ningún sujeto válido. Ajusta MIN_POS o revisa etiquetas.")

X_train = np.concatenate(X_train_list, axis=0).astype(np.float32, copy=False)
y_train = np.concatenate(y_train_list, axis=0).astype(np.int64, copy=False)

print("-"*60)
print("[INFO] DATASET FINAL")
print("-"*60)
print(f"[INFO] Sujetos usados: {subjects_aceptados}")
print(f"[INFO] Tamaño X_train: {X_train.shape} | dtype={X_train.dtype}")
print(f"[INFO] Tamaño y_train: {y_train.shape} | dtype={y_train.dtype}")
print(f"[INFO] Positivos: {np.sum(y_train==1)}")
print(f"[INFO] Negativos: {np.sum(y_train==0)}")
print("-"*60)

# 4. Guardar
np.save(IMPUT_RESULTS_DIR / "X_train.npy", X_train)
np.save(IMPUT_RESULTS_DIR / "y_train.npy", y_train)

print("[OK] Dataset guardado correctamente.")
print("Fin.")
