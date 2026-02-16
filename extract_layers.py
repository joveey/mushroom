# -*- coding: utf-8 -*-
"""
Extract intermediate layer outputs from Keras model
(using backend.function to avoid graph disconnected issue)
"""

import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

# ===============================
# CONFIG
# ===============================
RUN_DIR = Path("runs/mushroom_v1")
KERAS_MODEL_DIR = RUN_DIR / "keras_model_for_analysis"

DATASET_DIR = Path(r"C:\Users\Jovi\Downloads\splitted_dataset")
IMAGE_DIR = DATASET_DIR / "test" / "edible"
IMG_SIZE = 224

# ===============================
# LOAD MODEL
# ===============================
model = tf.keras.models.load_model(KERAS_MODEL_DIR)
print("✔ Keras model loaded from:", KERAS_MODEL_DIR)

# ===============================
# LIST MODEL LAYERS
# ===============================
print("\n=== MODEL LAYERS ===")
for i, layer in enumerate(model.layers):
    print(f"{i:02d}", layer.name, layer.output_shape)

# ===============================
# LOAD IMAGE
# ===============================
img_path = list(IMAGE_DIR.glob("*.jpg"))[0]
print("\nUsing image:", img_path.name)

img = cv2.imread(str(img_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

print("Input shape:", img.shape)

# ===============================
# BACKEND FUNCTIONS (KEY FIX)
# ===============================
get_conv = tf.keras.backend.function(
    [model.input],
    [model.get_layer("MobilenetV3small").output]
)

get_gap = tf.keras.backend.function(
    [model.input],
    [model.get_layer("global_average_pooling2d").output]
)

get_logits = tf.keras.backend.function(
    [model.input],
    [model.get_layer("logits").output]
)

# ===============================
# RUN INFERENCE
# ===============================
conv_out = get_conv([img])[0]
pool_out = get_gap([img])[0]
logits = get_logits([img])[0]
softmax = tf.nn.softmax(logits).numpy()

# ===============================
# PRINT OUTPUTS
# ===============================
print("\n=== CONVOLUTION (BACKBONE FEATURE MAP) ===")
print("Shape:", conv_out.shape)
print("Sample (3x3, channel 0):")
print(conv_out[0, :3, :3, 0])

print("\n=== GLOBAL AVERAGE POOLING OUTPUT ===")
print("Shape:", pool_out.shape)
print(pool_out[0][:8])

print("\n=== DENSE (LOGITS) OUTPUT ===")
print(logits)

print("\n=== SOFTMAX OUTPUT ===")
print(softmax)