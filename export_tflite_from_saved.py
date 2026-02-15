import argparse
from pathlib import Path
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--saved_dir", required=True)
ap.add_argument("--out_dir", required=True)
args = ap.parse_args()

saved = Path(args.saved_dir)
out   = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

# float32
conv = tf.lite.TFLiteConverter.from_saved_model(str(saved))
(out/"model_float32_calib.tflite").write_bytes(conv.convert())

# dynamic range
conv = tf.lite.TFLiteConverter.from_saved_model(str(saved))
conv.optimizations = [tf.lite.Optimize.DEFAULT]
(out/"model_dynamic_calib.tflite").write_bytes(conv.convert())

print("Export OK ->", out.resolve())
