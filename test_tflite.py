# -*- coding: utf-8 -*-
r"""
Tes inferensi TFLite + keputusan safety-first (EDIBLE?/POISONOUS/ABSTAIN)

Contoh:
python test_tflite.py --run_dir runs\mushroom_v1 --image "C:\Users\Jovi\Downloads\splitted_dataset\test\poisonous\FILE.jpg"
"""
import argparse, json, time
from pathlib import Path
import numpy as np, cv2, tensorflow as tf

def load_image(path: Path, size):
    img = cv2.imread(str(path))
    if img is None: raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size[1], size[0])).astype(np.float32)  # [0,255]
    return img

def run_tflite(model_path: Path, img_np):
    itp = tf.lite.Interpreter(model_path=str(model_path)); itp.allocate_tensors()
    inp = itp.get_input_details()[0]; out = itp.get_output_details()[0]
    x = img_np[None, ...]
    if inp["dtype"].__name__ == "int8":
        scale, zero = inp["quantization"]; x = (x/scale + zero).round().astype(np.int8)
    else:
        x = x.astype(np.float32)
    t0 = time.time(); itp.set_tensor(inp["index"], x); itp.invoke()
    y = itp.get_tensor(out["index"])[0]; ms = (time.time()-t0)*1000
    if out["dtype"].__name__ == "int8":
        scale, zero = out["quantization"]; y = (y.astype(np.float32)-zero)*scale
    return y.astype(np.float32), ms

def decide(probs, classes, tau_ed=0.90, tau_po=0.60):
    idx_ed = classes.index("edible") if "edible" in classes else 0
    idx_po = classes.index("poisonous") if "poisonous" in classes else 1
    p_ed, p_po = float(probs[idx_ed]), float(probs[idx_po])
    if p_ed >= tau_ed and p_ed > p_po: return "EDIBLE? (baca peringatan)"
    if p_po >= tau_po and p_po > p_ed: return "POISONOUS — JANGAN KONSUMSI"
    return "ABSTAIN — Perlu verifikasi"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--model", type=str, default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = json.load(open(run_dir/"config.json"))
    classes = cfg["class_names"]; H, W = cfg["img_size"]
    tau_ed, tau_po = cfg["tau_edible"], cfg["tau_poison"]

    model_path = Path(args.model) if args.model else (
        run_dir/"model_dynamic_calib.tflite"
        if (run_dir/"model_dynamic_calib.tflite").exists()
        else run_dir/"model_float32_calib.tflite"
    )

    img_np = load_image(Path(args.image), (H, W))
    probs, ms = run_tflite(model_path, img_np)
    probs = probs / (probs.sum() + 1e-9)

    print("Model   :", model_path.name)
    print("Classes :", classes)
    print("Probs   :", {c: round(float(p),4) for c,p in zip(classes, probs)})
    print("Decision:", decide(probs, classes, tau_ed, tau_po))
    print("Latency :", round(ms,2), "ms")

if __name__ == "__main__":
    main()
