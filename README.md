# Mushroom Local Pack (Safety-First, TFLite, Windows-friendly)

Pipeline lokal untuk klasifikasi jamur **edible vs poisonous** (TensorFlow) + **temperature calibration** + ekspor **SavedModel** dan **TFLite** (float32, dynamic range, opsional INT8) dengan keputusan **Safety-First** (bisa **ABSTAIN**).

## Isi repo (utama)
- `requirements.txt` – dependencies
- `train_local.py` – training MobileNetV3Small (2 tahap) + temperature scaling + export SavedModel/TFLite
- `evaluate.py` – evaluasi SavedModel (classification report, confusion matrix, ECE)
- `test_tflite.py` – inferensi 1 gambar via TFLite + keputusan safety-first
- `test_10_citra.py` – inferensi hingga 10 gambar (print tabel, opsional simpan CSV)
- `clean_dataset_tf.py` – bersihkan dataset supaya aman untuk `image_dataset_from_directory`
- `export_tflite_from_saved.py` – export TFLite dari folder SavedModel yang sudah ada

## Struktur dataset yang diharapkan
Script mengasumsikan root folder dengan layout:
```
<DATA_ROOT>/
  train/
    edible/
    poisonous/
  val/
    edible/
    poisonous/
  test/
    edible/
    poisonous/
```
Repo ini sudah punya contoh folder `splitted_dataset/` (dan contoh gambar `sample_10/`).

## Quickstart (PowerShell / Windows)
```ps1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
```

## (Opsional) bersihkan dataset
Kalau ada file gambar yang korup / tidak bisa didecode TensorFlow:
```ps1
python clean_dataset_tf.py .\splitted_dataset
```
Output log masalah ada di `bad_files_tf.txt`.

## Train + export
Catatan: run pertama akan mencoba download bobot ImageNet untuk MobileNetV3Small (butuh internet jika belum cache).
```ps1
python train_local.py --data_root .\splitted_dataset --out_dir runs\mushroom_v1
```
Opsi penting:
- `--epochs1`, `--epochs2` (default 5/5)
- `--no_int8` untuk skip export INT8

## Evaluasi (SavedModel)
Pastikan `--img_size` sama seperti saat training (default 224).
```ps1
python evaluate.py --run_dir runs\mushroom_v1 --data_root .\splitted_dataset
```

## Inferensi TFLite (1 gambar)
```ps1
python test_tflite.py --run_dir runs\mushroom_v1 --image .\splitted_dataset\test\edible\<FILE>.jpg
```
Default model: `model_dynamic_calib.tflite` (kalau ada), fallback ke `model_float32_calib.tflite`. Bisa override via `--model`.

## Inferensi TFLite (hingga 10 gambar)
```ps1
python test_10_citra.py --run_dir runs\mushroom_v1 --images_dir .\sample_10 --out_csv hasil_10_citra.csv
```

## Output/artifacts (di `--out_dir`)
- `config.json` (temperature + threshold safety-first + metadata)
- `labels.txt`
- `saved_model_calib/`
- `model_float32_calib.tflite`
- `model_dynamic_calib.tflite`
- `model_int8_calib.tflite` (kalau tidak pakai `--no_int8`)

## Catatan safety-first
- Threshold default ada di `config.json`: `tau_edible=0.90`, `tau_poison=0.60`.
- `test_tflite.py` / `test_10_citra.py` akan mengembalikan **ABSTAIN** jika confidence belum melewati threshold yang sesuai.
