"""(OPSIONAL) Fine-tune IndoBERT dan tambahkan hasilnya ke artefak evaluasi.

Disarankan dijalankan di Google Colab dengan GPU (Runtime > Change runtime type > T4).
Jalankan dari root proyek SETELAH `python -m src.train`:

    pip install -r requirements-indobert.txt
    python scripts/train_indobert.py

Skrip ini memakai split data yang SAMA dengan model klasik (seed tetap), lalu:
  - menambah kolom pred_indobert & proba_spam_indobert ke artifacts/test_predictions.csv
  - menambah entri "indobert" ke artifacts/metrics.json
sehingga IndoBERT otomatis muncul di halaman Evaluasi dan Analisis Kesalahan.
Model IndoBERT sendiri disimpan di artifacts/indobert/ (tidak dipakai oleh aplikasi web).

CATATAN: skrip ini belum diuji di lingkungan pengembangan proyek (membutuhkan GPU dan
akses ke Hugging Face). Jika ada error karena beda versi library, sesuaikan
argumen TrainingArguments.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                          TrainingArguments)

from src.config import ARTIFACTS, LABELS, POS_LABEL, RANDOM_SEED
from src.data import deduplicate, load_dataset, split
from src.evaluation import metrics_block

MODEL_ID = "indobenchmark/indobert-base-p2"
MAX_LEN = 128
EPOCHS = 3
BATCH = 16
LR = 2e-5
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}     # Normal=0, Spam=1
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}


class SmsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN, padding=True)
        self.labels = [LABEL2ID[l] for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def main():
    # Split identik dengan src/train.py
    df, _ = deduplicate(load_dataset())
    train, test = split(df)

    preds_path = ARTIFACTS / "test_predictions.csv"
    if not preds_path.exists():
        sys.exit("Jalankan dulu: python -m src.train")
    preds = pd.read_csv(preds_path)
    assert preds["text"].tolist() == test["text"].tolist(), \
        "Data uji tidak sama dengan test_predictions.csv. Jalankan ulang `python -m src.train`."

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)

    args = TrainingArguments(
        output_dir=str(ARTIFACTS / "indobert_tmp"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=64,
        learning_rate=LR,
        weight_decay=0.01,
        seed=RANDOM_SEED,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=SmsDataset(train["text"], train["label"], tok))

    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0

    test_ds = SmsDataset(test["text"], test["label"], tok)
    t0 = time.time()
    logits = trainer.predict(test_ds).predictions
    infer_ms = (time.time() - t0) / len(test) * 1000     # batch, bukan satu-satu

    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    p_spam = probs[:, LABEL2ID[POS_LABEL]]
    y_pred = np.array([ID2LABEL[i] for i in probs.argmax(1)])
    y_true = test["label"].to_numpy()

    m = metrics_block(y_true, y_pred, p_spam)
    m.update({"name": "IndoBERT (fine-tuned)",
              "best_params": {"epochs": EPOCHS, "lr": LR, "batch": BATCH, "max_len": MAX_LEN},
              "cv_f1_macro": None, "train_seconds": round(train_s, 1),
              "infer_ms": round(infer_ms, 2)})
    print({k: round(v, 4) for k, v in m.items() if isinstance(v, float)})

    preds["pred_indobert"] = y_pred
    preds["proba_spam_indobert"] = np.round(p_spam, 4)
    preds.to_csv(preds_path, index=False)

    mpath = ARTIFACTS / "metrics.json"
    metrics = json.loads(mpath.read_text())
    metrics["models"]["indobert"] = m
    mpath.write_text(json.dumps(metrics, indent=2))

    out = ARTIFACTS / "indobert"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print("Selesai. Hasil ditambahkan ke metrics.json dan test_predictions.csv")


if __name__ == "__main__":
    main()
