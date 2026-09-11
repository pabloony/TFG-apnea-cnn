# cnn_model.py
# Arquitectura Piorecky-style para detección de apnea/desaturación
# Keras 2.3.1 / TensorFlow 2.x backend

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics
from .config import DROPOUT_CONV, DROPOUT_DENSE,INPUT_SHAPE

def build_cnn(input_shape=INPUT_SHAPE): #este es el modelo propuesto por Piorecky , puede que haga modificaciones 
    """
    Crea el modelo CNN con entrada (2, 500, 1):
    - Conv2D(174, kernel 1x1) + ReLU
    - MaxPool(1x2) + Dropout(0.6)
    - Conv2D(308, kernel 2x15) + ReLU
    - MaxPool(1x2) + Dropout(0.6)
    - Conv2D(96, kernel 1x45) + ReLU
    - MaxPool(1x2) + Dropout(0.6)
    - Flatten
    - Dense(148) + ReLU + Dropout(0.6)
    - Dense(86)  + ReLU + Dropout(0.6)
    - Dense(1)   + Sigmoid
    """
    model = models.Sequential(name="ApnoeCNN")

    # Capa 1: Conv2D con 174 filtros y kernel 1x1 (equivale a mezcla/ponderación por canal y tiempo)
    model.add(layers.Conv2D(
        filters=174, kernel_size=(1, 1), strides=(1, 1), padding='valid',
        activation='relu', input_shape=input_shape, name='conv1_k1x1_f174'))
    # Salida esperada: (2, 500, 174). Tabla 12. :contentReference[oaicite:2]{index=2}

    # Reduce a la mitad el eje temporal (500 -> 250) manteniendo el eje de “canales” vertical (2)
    model.add(layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), name='pool1'))
    # Regularización fuerte para evitar overfitting (valor óptimo 0.6 en el paper)
    model.add(layers.Dropout(DROPOUT_CONV, name='drop1'))  # :contentReference[oaicite:3]{index=3}

    # Capa 2: Conv2D con kernel 2x15 y 308 filtros
    model.add(layers.Conv2D(
        filters=308, kernel_size=(2, 15), strides=(1, 1), padding='valid',
        activation='relu', name='conv2_k2x15_f308'))
    # Al usar kernel alto=2, comprimimos (2 -> 1) el eje “canales de entrada”; temporal 250->236.
    # Tamaño esperado: (1, 236, 308). Tabla 12. :contentReference[oaicite:4]{index=4}
    model.add(layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), name='pool2'))  # (1,118,308)
    model.add(layers.Dropout(DROPOUT_CONV, name='drop2'))

    # Capa 3: Conv2D con kernel 1x45 y 96 filtros
    model.add(layers.Conv2D(
        filters=96, kernel_size=(1, 45), strides=(1, 1), padding='valid',
        activation='relu', name='conv3_k1x45_f96'))
    # Esperado: (1, 74, 96) -> pool -> (1, 37, 96). Tabla 12. :contentReference[oaicite:5]{index=5}
    model.add(layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), name='pool3'))
    model.add(layers.Dropout(DROPOUT_CONV, name='drop3'))

    # Aplana mapas de características: 1*37*96 = 3552 características
    model.add(layers.Flatten(name='flatten'))

    # Bloque denso 1: 148 neuronas + ReLU
    model.add(layers.Dense(148, activation='relu', name='fc1_148'))  # Tabla 12. :contentReference[oaicite:6]{index=6}
    model.add(layers.Dropout(DROPOUT_DENSE, name='drop4'))

    # Bloque denso 2: 86 neuronas + ReLU
    model.add(layers.Dense(86, activation='relu', name='fc2_86'))  # Tabla 12. :contentReference[oaicite:7]{index=7}
    model.add(layers.Dropout(DROPOUT_DENSE, name='drop5'))

    # Salida binaria con sigmoide (probabilidad de evento)
    model.add(layers.Dense(1, activation='sigmoid', name='out_sigmoid'))

    return model
