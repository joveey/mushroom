import argparse, json, time, csv
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf


ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def load_image(path: Path, size):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size[1], size[0])).astype(np.float32)  # [0,255]
    return img


def run_tflite(model_path: Path, img_np):
    itp = tf.lite.Interpreter(model_path=str(model_path))
    itp.allocate_tensors()
    inp = itp.get_input_details()[0]
    out = itp.get_output_details()[0]

    x = img_np[None, ...]
    if inp["dtype"].__name__ == "int8":
        scale, zero = inp["quantization"]
        x = (x / scale + zero).round().astype(np.int8)
    else:
        x = x.astype(np.float32)

    t0 = time.time()
    itp.set_tensor(inp["index"], x)
    itp.invoke()
    y = itp.get_tensor(out["index"])[0]
    ms = (time.time() - t0) * 1000

    if out["dtype"].__name__ == "int8":
        scale, zero = out["quantization"]
        y = (y.astype(np.float32) - zero) * scale

    return y.astype(np.float32), ms


def decide(probs, classes, tau_ed=0.90, tau_po=0.60):
    # Mengikuti pola punyamu, aman walau urutan class_names beda
    idx_ed = classes.index("edible") if "edible" in classes else 0
    idx_po = classes.index("poisonous") if "poisonous" in classes else 1

    p_ed = float(probs[idx_ed])
    p_po = float(probs[idx_po])

    if p_ed >= tau_ed and p_ed > p_po:
        return "EDIBLE? (baca peringatan)"
    if p_po >= tau_po and p_po > p_ed:
        return "POISONOUS - JANGAN KONSUMSI"
    return "ABSTAIN - Perlu verifikasi"


def collect_images(images_dir: Path):
    files = []
    if images_dir.is_file():
        # Kalau user kasih 1 file langsung
        if images_dir.suffix.lower() in ALLOWED:
            return [images_dir]
        return []

    # Ambil rekursif biar bisa langsung arah ke folder test
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED:
            files.append(p)

    # Sort biar konsisten
    files.sort(key=lambda x: x.name.lower())
    return files


def infer_10(run_dir: Path, images_dir: Path, model_override: str = "", out_csv: str = ""):
    cfg = json.load(open(run_dir / "config.json", encoding="utf-8"))
    classes = cfg["class_names"]
    H, W = cfg["img_size"]
    tau_ed = cfg.get("tau_edible", 0.90)
    tau_po = cfg.get("tau_poison", 0.60)

    # Pilih model default seperti punyamu
    if model_override:
        model_path = Path(model_override)
    else:
        dyn = run_dir / "model_dynamic_calib.tflite"
        flt = run_dir / "model_float32_calib.tflite"
        model_path = dyn if dyn.exists() else flt

    imgs = collect_images(images_dir)[:10]

    if not imgs:
        print("Tidak ada file gambar yang ditemukan.")
        return

    rows = []
    for i, img_path in enumerate(imgs, start=1):
        img_np = load_image(img_path, (H, W))
        probs, ms = run_tflite(model_path, img_np)
        probs = probs / (probs.sum() + 1e-9)

        # Ambil prob sesuai nama kelas, versi aman untuk Pylance
        p_map = {c: float(p) for c, p in zip(classes, probs)}

        p_ed = float(p_map.get("edible", float(probs[0])))

        default_po = float(probs[1]) if len(probs) > 1 else 0.0
        p_po = float(p_map.get("poisonous", default_po))

        keputusan_model = classes[int(np.argmax(probs))].upper()
        keputusan_sf = decide(probs, classes, tau_ed, tau_po)

        rows.append({
            "No": i,
            "Nama Citra": img_path.name,
            "Path": str(img_path),
            "Prob Edible": round(float(p_ed), 4),
            "Prob Poisonous": round(float(p_po), 4),
            "Keputusan Model": keputusan_model,
            "Keputusan Safety-first": keputusan_sf,
            "Latency ms": round(float(ms), 2),
        })

    # Print ringkas buat kamu copas ke Tabel 4.1
    print("Model   :", model_path.name)
    print("Classes :", classes)
    print("Tau     :", {"tau_edible": tau_ed, "tau_poison": tau_po})
    print()

    header = [
        "No", "Nama Citra", "Prob Edible", "Prob Poisonous",
        "Keputusan Model", "Keputusan Safety-first", "Latency ms"
    ]
    print(" | ".join(header))
    print("-" * 120)
    for r in rows:
        print(
            f'{r["No"]} | {r["Nama Citra"]} | {r["Prob Edible"]} | {r["Prob Poisonous"]} | '
            f'{r["Keputusan Model"]} | {r["Keputusan Safety-first"]} | {r["Latency ms"]}'
        )

    # Optional save CSV
    if out_csv:
        out_path = Path(out_csv)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\nCSV saved:", out_path.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--out_csv", type=str, default="")
    args = ap.parse_args()

    infer_10(
        run_dir=Path(args.run_dir),
        images_dir=Path(args.images_dir),
        model_override=args.model or "",
        out_csv=args.out_csv or ""
    )


if __name__ == "__main__":
    main()
