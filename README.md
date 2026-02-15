# Mushroom Local Pack (Safety-First, TFLite, Windows-friendly)

This pack gives you a **complete local pipeline** to train a mushroom classifier from the Kaggle dataset and export it to **TFLite** with **calibrated probabilities** and **Safety-First decision thresholds** (with **ABSTAIN**).

## Files
- `requirements.txt` – minimal dependencies
- `train_local.py` – train + temperature calibration + export TFLite (float32, dynamic, and optional INT8)
- `test_tflite.py` – run a quick inference on a single image with Safety-First decision

## Quickstart (PowerShell on Windows)
```ps1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
```

Download the dataset (Kaggle CLI):
```ps1
# Place kaggle.json at C:\Users\<YOU>\.kaggle\kaggle.json
mkdir data\mushrooms
kaggle datasets download -d benedictusjason/edible-and-poisonous-mushroom-classification -p data\mushrooms --unzip
```

Train locally:
```ps1
python train_local.py --data_dir data\mushrooms --out_dir runs\mushroom_v1
```

Test inference (replace with a real image path):
```ps1
python test_tflite.py --run_dir runs\mushroom_v1 --image data\mushrooms\edible\some_image.jpg
```

## Notes
- The exported models named `model_*_calib.tflite` are **calibrated** (softmax over logits/T).
- Safety thresholds are in `config.json` (`tau_edible=0.90`, `tau_poison=0.60`). Adjust if needed after seeing validation metrics.
- For smallest size, enable INT8 (default ON). If you want to skip INT8, pass `--no_int8` to the training script.
