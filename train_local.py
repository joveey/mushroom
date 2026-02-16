# -*- coding: utf-8 -*-
r"""
Train MobileNetV3Small on Kaggle mushroom (edible/poisonous) dengan layout:
C:\Users\Jovi\Downloads\splitted_dataset\
  train\edible|poisonous
  val\edible|poisonous
  test\edible|poisonous

Fitur:
- Augment ringan + freeze->unfreeze 2 tahap
- Temperature calibration (safety-first)
- Export TFLite (float32, dynamic, INT8)
- Quick eval di VAL/TEST

Jalankan contoh:
python train_local.py --data_root "C:\Users\Jovi\Downloads\splitted_dataset" --out_dir runs\mushroom_v1
"""
import os, json, time, argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


# ---------- Utils ----------
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def ds_from_dir(dir_path: Path, img_size=(224, 224), batch=32, shuffle=True):
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    return tf.keras.utils.image_dataset_from_directory(
        dir_path, image_size=img_size, batch_size=batch, shuffle=shuffle
    )

def build_datasets_with_splits(train_dir: Path, val_dir: Path, test_dir: Optional[Path],
                               img_size=(224,224), batch=32):
    at = tf.data.AUTOTUNE
    train_ds = ds_from_dir(train_dir, img_size, batch, shuffle=True).shuffle(1024).prefetch(at)
    val_ds   = ds_from_dir(val_dir,   img_size, batch, shuffle=False).prefetch(at)
    test_ds  = None
    if test_dir and test_dir.exists():
        test_ds = ds_from_dir(test_dir, img_size, batch, shuffle=False).prefetch(at)
    # kelas mengikuti isi train dir
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    return train_ds, val_ds, test_ds, class_names

def build_model(num_classes: int, img_size=(224,224)):
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=img_size+(3,), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inp = tf.keras.Input(shape=img_size+(3,), name="input_image")
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inp)
    x = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomContrast(0.10),
    ], name="augment")(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    logits = tf.keras.layers.Dense(num_classes, activation=None, name="logits")(x)  # from_logits
    model = tf.keras.Model(inp, logits, name="mushroom_cnn")
    return model, base

def train_two_stage(model, base, train_ds, val_ds, epochs1=5, epochs2=5):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=2, mode="max", restore_best_weights=True)
    ]

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=loss_fn, metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])
    log("Stage-1: backbone frozen")
    model.fit(train_ds, epochs=epochs1, validation_data=val_ds, callbacks=cbs)

    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=loss_fn, metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])
    log("Stage-2: unfreeze last ~30 layers")
    model.fit(train_ds, epochs=epochs2, validation_data=val_ds, callbacks=cbs)

def collect_val_logits(model, val_ds) -> Tuple[np.ndarray, np.ndarray]:
    logits_val, y_val = [], []
    for xb, yb in val_ds:
        z = model.predict(xb, verbose=0)  # logits
        logits_val.append(z); y_val.append(yb.numpy())
    return np.vstack(logits_val), np.hstack(y_val)

def temperature_scaling(logits_val: np.ndarray, y_val: np.ndarray, steps=400, lr=0.05) -> float:
    """Optimisasi logT (agar T>0) untuk meminimasi NLL pada val."""
    logits_tf = tf.convert_to_tensor(logits_val, dtype=tf.float32)
    y_true = tf.convert_to_tensor(y_val, dtype=tf.int32)
    logT = tf.Variable(0.0, dtype=tf.float32)  # T = exp(logT)
    opt = tf.keras.optimizers.Adam(lr)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            T = tf.exp(logT)
            scaled = logits_tf / T
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y_true, tf.nn.softmax(scaled))
            )
        opt.apply_gradients([(tape.gradient(loss, logT), logT)])
    return float(tf.exp(logT).numpy())

def build_prob_models(model_logits: tf.keras.Model, temperature: float, img_size=(224,224)):
    inp = tf.keras.Input(shape=img_size+(3,), name="input_image")
    logits = model_logits(inp)
    probs_uncalib = tf.nn.softmax(logits, name="softmax_uncalib")
    logits_scaled = tf.keras.layers.Lambda(lambda z: z / temperature, name="divide_by_T")(logits)
    probs_calib = tf.nn.softmax(logits_scaled, name="softmax_calibrated")
    return tf.keras.Model(inp, probs_uncalib), tf.keras.Model(inp, probs_calib)

def export_tflite(saved_dir: Path, out_path: Path, dynamic=False, int8=False, rep_ds=None):
    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
    if int8:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        if rep_ds is None:
            raise ValueError("INT8 export requires representative dataset.")
        conv.representative_dataset = rep_ds
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    elif dynamic:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    tfl = conv.convert()
    out_path.write_bytes(tfl)

def make_rep_ds(val_ds, limit_batches=100):
    n = 0
    for xb, _ in val_ds:
        yield [xb.numpy().astype(np.float32)]  # model sudah handle preprocess di dalam
        n += 1
        if n >= limit_batches: break

def quick_eval(saved_model_dir: Path, ds, title: str):
    model = tf.saved_model.load(str(saved_model_dir))
    infer = model.signatures.get("serving_default")
    total, correct = 0, 0
    for xb, yb in ds:
        out = list(infer(input_image=xb).values())[0].numpy()
        pred = out.argmax(1); y = yb.numpy()
        total += len(y); correct += (pred == y).sum()
    acc = correct / max(total, 1)
    print(f"== {title} acc: {acc:.4f} ({total} samples)")
    return acc


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True,
                    help="Folder yang berisi train/ val/ test/")
    ap.add_argument("--out_dir", type=str, default="runs/mushroom_v1")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs1", type=int, default=5)
    ap.add_argument("--epochs2", type=int, default=5)
    ap.add_argument("--no_int8", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_root)
    train_dir = root / "train"
    val_dir   = root / "val"
    test_dir  = root / "test"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("TensorFlow", tf.__version__)
    print("Train:", train_dir); print("Val:", val_dir); print("Test:", test_dir)

    train_ds, val_ds, test_ds, class_names = build_datasets_with_splits(
        train_dir, val_dir, test_dir, img_size=(args.img_size,args.img_size), batch=args.batch
    )
    print("Classes:", class_names)

    # Build + Train
    model_logits, base = build_model(num_classes=len(class_names), img_size=(args.img_size,args.img_size))
    train_two_stage(model_logits, base, train_ds, val_ds, args.epochs1, args.epochs2)

    # Calibration
    log("Collecting validation logits")
    logits_val, y_val = collect_val_logits(model_logits, val_ds)
    log("Fitting temperature (calibration)")
    T_calib = temperature_scaling(logits_val, y_val, steps=400, lr=0.05)
    print("Temperature =", T_calib)

    # Probability models
    _, prob_calib = build_prob_models(model_logits, T_calib, img_size=(args.img_size,args.img_size))

    # ===============================
    # SAVE KERAS MODEL (FOR ANALYSIS ONLY)
    # ===============================
    keras_analysis_dir = out_dir / "keras_model_for_analysis"
    model_logits.save(keras_analysis_dir)
    print("Saved Keras model for analysis at:", keras_analysis_dir)

    # Save labels & config (threshold safety-first)
    (out_dir / "labels.txt").write_text("\n".join(class_names), encoding="utf-8")
    cfg = {"temperature": float(T_calib),
           "tau_edible": 0.90, "tau_poison": 0.60,
           "class_names": class_names, "img_size": [args.img_size, args.img_size]}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Export SavedModel (calibrated) + Quick eval
    saved_calib = out_dir / "saved_model_calib"
    tf.saved_model.save(prob_calib, str(saved_calib))
    quick_eval(saved_calib, val_ds, "VAL")
    if test_ds is not None: quick_eval(saved_calib, test_ds, "TEST")

    # Export TFLite
    export_tflite(saved_calib, out_dir / "model_float32_calib.tflite")
    export_tflite(saved_calib, out_dir / "model_dynamic_calib.tflite", dynamic=True)
    if not args.no_int8:
        rep = lambda: make_rep_ds(val_ds, limit_batches=100)
        export_tflite(saved_calib, out_dir / "model_int8_calib.tflite", int8=True, rep_ds=rep)
    else:
        print("Skipping INT8 export.")

    print("Done. Artifacts in", out_dir.resolve())

if __name__ == "__main__":
    main()
