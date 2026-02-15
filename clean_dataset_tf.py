# -*- coding: utf-8 -*-
r"""
Bersihkan dataset agar aman untuk image_dataset_from_directory (decoder TensorFlow).
- Mencoba decode dengan tf.io.decode_image
- Jika gagal -> re-encode ke JPG RGB standar (Pillow)
- Jika masih gagal -> hapus file
- Log masalah ke bad_files_tf.txt

Cara pakai:
python clean_dataset_tf.py "C:\Users\Jovi\Downloads\splitted_dataset"
# atau tanpa argumen (pakai DEFAULT_ROOT)
"""

from pathlib import Path
from PIL import Image
import tensorflow as tf
import sys

# ====== EDIT default path kalau mau hardcode ======
DEFAULT_ROOT = Path(r"C:\Users\Jovi\Downloads\splitted_dataset")
# ===================================================

ROOT = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_ROOT
ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

log_bad = []
converted = 0
deleted = 0
checked = 0

def tf_can_decode(p: Path) -> bool:
    try:
        raw = tf.io.read_file(str(p))
        img = tf.io.decode_image(raw, channels=3)  # decoder yang sama dipakai Keras
        _ = img.shape  # trigger decode
        return True
    except Exception as e:
        log_bad.append(f"{p} :: tf-decode-failed ({e})")
        return False

def reencode_to_jpg(p: Path) -> Path | None:
    """Re-encode ke JPG RGB (overwrite kalau sudah .jpg). Return path output, None jika gagal."""
    global converted, deleted
    try:
        with Image.open(p) as im:
            im.load()
            im = im.convert("RGB")
        out = p.with_suffix(".jpg")
        # kalau sudah .jpg, normalize dengan save ulang
        im.save(out, "JPEG", quality=95, optimize=True)
        if out != p and p.exists():
            try:
                p.unlink()
                deleted += 1
            except:
                pass
        converted += 1
        return out
    except Exception as e:
        log_bad.append(f"{p} :: reencode-failed ({e})")
        try:
            p.unlink()
            deleted += 1
        except:
            pass
        return None

def process_file(p: Path):
    global checked, deleted
    checked += 1
    if p.stat().st_size == 0:
        log_bad.append(f"{p} :: zero-byte")
        try:
            p.unlink()
            deleted += 1
        except:
            pass
        return

    # 1) coba decode apa adanya
    if tf_can_decode(p):
        return

    # 2) kalau gagal -> re-encode ke JPG
    out = reencode_to_jpg(p)
    if out is None:
        return

    # 3) cek ulang dengan TF
    if not tf_can_decode(out):
        log_bad.append(f"{out} :: still-fails-after-reencode")
        try:
            out.unlink()
            deleted += 1
        except:
            pass

def is_image_file(p: Path) -> bool:
    if not p.is_file():
        return False
    # lewati file hidden/system umum
    if p.name.startswith(".") or p.name.lower() in {"thumbs.db", "desktop.ini"}:
        return False
    return True

def main():
    if not ROOT.exists():
        print(f"[ERROR] Path tidak ditemukan: {ROOT}")
        sys.exit(1)

    for split in ["train", "val", "test"]:
        d = ROOT / split
        if not d.exists():
            print(f"[WARN] Folder split tidak ada: {d}")
            continue
        for p in d.rglob("*"):
            if not is_image_file(p):
                continue
            process_file(p)

    Path("bad_files_tf.txt").write_text("\n".join(log_bad), encoding="utf-8")
    print(f"Checked  : {checked} files")
    print(f"Converted: {converted} files")
    print(f"Deleted  : {deleted} files")
    print(f"Problems : {len(log_bad)} (lihat bad_files_tf.txt)")
    print("Done.")

if __name__ == "__main__":
    main()
