import numpy as np
from pathlib import Path
from .cnn_model import build_cnn
import tensorflow as tf

# ── Importar rutas y parámetros desde config ──────────────────────────
from .config import (
    BATCH_SIZE, EPOCHS, LR,
    DROPOUT_CONV, DROPOUT_DENSE,  
    IMPUT_DATA_DIR, TRAIN_RESULTS_DIR,
    X_TRAIN_PATH, Y_TRAIN_PATH, X_VAL_PATH, Y_VAL_PATH,
    MODEL_BEST_PATH, MODEL_FINAL_PATH, HISTORY_CSV_PATH,
    USE_REDUCE_LR, REDUCE_LR_FACTOR, REDUCE_LR_PATIENCE, REDUCE_LR_MIN_LR,
    USE_EARLY_STOP, EARLY_STOP_PATIENCE,INPUT_SHAPE
)

DATA_DIR = Path(IMPUT_DATA_DIR)
out_dir  = Path(TRAIN_RESULTS_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

# ── Verificar GPU ─────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
print(f"[INFO] GPUs disponibles: {gpus}")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ── Cargar datos ──────────────────────────────────────────
print("[INFO] Cargando datos...")
X_train = np.load(X_TRAIN_PATH, mmap_mode="r")
y_train = np.load(Y_TRAIN_PATH, mmap_mode="r").astype(np.float32)
X_val   = np.load(X_VAL_PATH)
y_val   = np.load(Y_VAL_PATH).astype(np.float32)

# FIX: n_train tiene que existir ANTES de definir gen_batches, porque
# gen_batches la usa dentro de su propio cuerpo (closure). Antes no estaba
# definida en ningun sitio -> NameError al ejecutar el generador.
n_train = X_train.shape[0]
steps_per_epoch = int(np.ceil(n_train / BATCH_SIZE))

print(f"[INFO] X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"[INFO] X_val  : {X_val.shape}   | y_val  : {y_val.shape}")
print(f"[INFO] steps_per_epoch: {steps_per_epoch}")

# ── Pipeline tf.data (solo para train) ───────────────────
def gen_batches():
    for start in range(0, n_train, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_train)
        Xb = np.asarray(X_train[start:end], dtype=np.float32)
        yb = np.asarray(y_train[start:end], dtype=np.float32)
        yield Xb, yb

# FIX BUG "epocas fantasma": antes el dataset era finito (se agotaba justo
# al final de cada epoca real) y al pasar steps_per_epoch explicito, Keras
# intentaba sacar batches del MISMO iterador ya agotado antes de crear uno
# nuevo -> una epoca de cada dos salia con 0 batches procesados (train
# metrics en 0, val_auc clavado igual que la epoca anterior). Con .repeat()
# el dataset es infinito y nunca se agota a mitad de nada; es steps_per_epoch
# quien marca donde corta cada epoca, sin StopIteration de por medio.
dataset = tf.data.Dataset.from_generator(
    gen_batches,
    output_signature=(
        tf.TensorSpec(shape=(None, *INPUT_SHAPE), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    ),
).repeat().prefetch(tf.data.AUTOTUNE)

# ── Modelo ────────────────────────────────────────────────
input_shape = X_train.shape[1:]  # (2, 750, 1) con W=15s, automático
model = build_cnn(input_shape=input_shape)
print(f"[INFO] Dropout  conv={DROPOUT_CONV}  dense={DROPOUT_DENSE}") 
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
    ],
)

# ── Callbacks ─────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.CSVLogger(
        str(Path(HISTORY_CSV_PATH)), append=False
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(Path(MODEL_BEST_PATH)),
        monitor="val_auc",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
]

if USE_REDUCE_LR:
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        min_lr=REDUCE_LR_MIN_LR,
        mode="max",
        verbose=1,
    ))

if USE_EARLY_STOP:
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        mode="max",
        verbose=1,
    ))

# ── Training ──────────────────────────────────────────────
print("[INFO] Iniciando training...")
history = model.fit(
    dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=callbacks,
)

# ── Guardar modelo final ───────────────────────────────────
model.save(Path(MODEL_FINAL_PATH))
print(f"[OK] Modelos guardados en {out_dir}")