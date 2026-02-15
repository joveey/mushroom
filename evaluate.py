# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

def load_ds(dir_path: Path, img_size=(224,224), batch=32):
    return tf.keras.utils.image_dataset_from_directory(
        dir_path, image_size=img_size, batch_size=batch, shuffle=False
    )

def expected_calibration_error(probs, labels, n_bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        idx = (conf > bins[i]) & (conf <= bins[i+1])
        if not np.any(idx):
            continue
        ece += idx.mean() * abs(acc[idx].mean() - conf[idx].mean())
    return float(ece)

def run_eval(model, ds, classes, title):
    infer = model.signatures.get("serving_default")
    y_true, y_pred, p_list = [], [], []
    for xb, yb in ds:
        out_map = infer(input_image=xb)
        # fleksibel: kalau key "softmax_calibrated" nggak ada, ambil key pertama
        key = "softmax_calibrated" if "softmax_calibrated" in out_map else next(iter(out_map.keys()))
        out = out_map[key].numpy()
        p_list.append(out)
        y_true.append(yb.numpy())
        y_pred.append(out.argmax(1))
    probs = np.vstack(p_list)
    y_true = np.hstack(y_true)
    y_pred = np.hstack(y_pred)

    print(f"\n== {title} ==")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("ECE:", expected_calibration_error(probs, y_true))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    root = Path(args.data_root)
    classes = json.load(open(run_dir / "config.json"))["class_names"]
    model = tf.saved_model.load(str(run_dir / "saved_model_calib"))

    val_ds  = load_ds(root / "val",  (args.img_size, args.img_size), args.batch)
    test_ds = load_ds(root / "test", (args.img_size, args.img_size), args.batch)

    run_eval(model, val_ds,  classes, "VALIDATION")
    run_eval(model, test_ds, classes, "TEST")

if __name__ == "__main__":
    main()
