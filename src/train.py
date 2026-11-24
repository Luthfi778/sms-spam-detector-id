"""Training dan evaluasi semua model klasik.

Jalankan dari root proyek:
    python -m src.train

Keluaran (folder artifacts/):
    models/*.joblib          pipeline lengkap (preprocessing + TF-IDF + model)
    metrics.json             metrik evaluasi per model (data uji)
    test_predictions.csv     prediksi & probabilitas tiap model pada data uji
    robustness.csv           recall Spam pada pesan yang disamarkan
    robustness_examples.csv  contoh pesan asli vs disamarkan
    data_summary.json        ringkasan data (distribusi, duplikat, kata teratas)
    dataset_clean.csv        dataset bersih + kolom split (train/test)
"""
import json
import platform
import time
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import (ARTIFACTS, LABELS, MODEL_NAMES, MODELS_DIR, POS_LABEL,
                     RANDOM_SEED, TEST_SIZE)
from .data import deduplicate, load_dataset, split
from .evaluation import metrics_block
from .preprocessing import STOPWORDS, clean_text, naive_clean
from .robustness import PERTURBATIONS, perturb_all


def make_features(preprocessor=clean_text):
    """TF-IDF kata (1-2 gram) + TF-IDF karakter (3-5 gram, tahan typo)."""
    return FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=preprocessor, lowercase=False,
                                 ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=preprocessor, lowercase=False,
                                 analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 sublinear_tf=True, max_features=30000)),
    ])


def make_models():
    """key -> (estimator, grid hyperparameter untuk GridSearchCV)."""
    return {
        "nb": (MultinomialNB(), {"clf__alpha": [0.05, 0.1, 0.5, 1.0]}),
        "lr": (LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=RANDOM_SEED),
               {"clf__C": [1, 10, 100]}),
    }


def top_words(texts, k=15):
    """Kata dengan document frequency tertinggi (tanpa stopword)."""
    counter = Counter()
    for t in texts:
        counter.update(set(w for w in clean_text(t).split()
                           if w not in STOPWORDS and len(w) > 2))
    return [[w, c] for w, c in counter.most_common(k)]


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data ---------------------------------------------------------------
    raw = load_dataset()
    df, dedup_info = deduplicate(raw)
    train, test = split(df)
    print(f"Data mentah {dedup_info['n_raw']} -> bersih {dedup_info['n_clean']} "
          f"(duplikat dibuang: {dedup_info['n_duplicates']}, kosong: {dedup_info['n_empty']}, "
          f"label bentrok: {dedup_info['n_conflicting']})")
    print(f"Train {len(train)} | Test {len(test)}")

    pd.concat([train.assign(split="train"), test.assign(split="test")]).to_csv(
        ARTIFACTS / "dataset_clean.csv", index=False)

    summary = {
        "dedup": dedup_info,
        "n_train": len(train), "n_test": len(test),
        "counts_raw": raw["label"].value_counts().to_dict(),
        "counts_clean": df["label"].value_counts().to_dict(),
        "counts_train": train["label"].value_counts().to_dict(),
        "counts_test": test["label"].value_counts().to_dict(),
        "top_words": {lab: top_words(train.loc[train["label"] == lab, "text"])
                      for lab in LABELS},
    }
    json.dump(summary, open(ARTIFACTS / "data_summary.json", "w"), indent=2)
    # Versi library saat training: aplikasi memeriksa ini agar model tidak dimuat
    # di lingkungan dengan versi scikit-learn yang berbeda (penyebab AttributeError).
    json.dump({"scikit-learn": sklearn.__version__, "python": platform.python_version()},
              open(ARTIFACTS / "versions.json", "w"), indent=2)

    X_train, y_train = train["text"].tolist(), train["label"].to_numpy()
    X_test, y_test = test["text"].tolist(), test["label"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # 2. Training + evaluasi ------------------------------------------------
    results, preds = {}, test[["text", "label"]].copy()
    fitted = {}
    for key, (clf, grid) in make_models().items():
        print(f"\n== {MODEL_NAMES[key]} ==")
        pipe = Pipeline([("features", make_features()), ("clf", clf)])
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=1)
        t0 = time.time()
        search.fit(X_train, y_train)
        train_s = time.time() - t0
        best = search.best_estimator_
        fitted[key] = best
        joblib.dump(best, MODELS_DIR / f"{key}.joblib", compress=3)

        proba = best.predict_proba(X_test)[:, list(best.classes_).index(POS_LABEL)]
        y_pred = best.predict(X_test)

        # waktu inferensi per pesan (satu per satu, seperti di aplikasi)
        sample = X_test[:200]
        t0 = time.time()
        for t in sample:
            best.predict_proba([t])
        infer_ms = (time.time() - t0) / len(sample) * 1000

        m = metrics_block(y_test, y_pred, proba)
        m.update({
            "name": MODEL_NAMES[key],
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_f1_macro": float(search.best_score_),
            "train_seconds": round(train_s, 1),
            "infer_ms": round(infer_ms, 2),
        })
        results[key] = m
        preds[f"pred_{key}"] = y_pred
        preds[f"proba_spam_{key}"] = np.round(proba, 4)
        print(f"best={m['best_params']} cv_f1_macro={m['cv_f1_macro']:.4f}")
        print(f"test: acc={m['accuracy']:.4f} P={m['precision_spam']:.4f} "
              f"R={m['recall_spam']:.4f} F1={m['f1_spam']:.4f}")

    json.dump({
        "meta": {"seed": RANDOM_SEED, "test_size": TEST_SIZE, "labels": LABELS,
                 "pos_label": POS_LABEL, "n_train": len(train), "n_test": len(test)},
        "models": results,
    }, open(ARTIFACTS / "metrics.json", "w"), indent=2)
    preds.to_csv(ARTIFACTS / "test_predictions.csv", index=False)

    # 3. Uji ketahanan ------------------------------------------------------
    spam_texts = test.loc[test["label"] == POS_LABEL, "text"].tolist()
    # Pembanding (ablation): Logistic Regression dengan preprocessing naif
    # (tanpa normalisasi spasi/titik/leetspeak) untuk menunjukkan nilai tambah normalisasi.
    best_c = results["lr"]["best_params"]["C"]
    naive_lr = Pipeline([
        ("features", make_features(naive_clean)),
        ("clf", LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced",
                                   random_state=RANDOM_SEED)),
    ]).fit(X_train, y_train)
    eval_models = {MODEL_NAMES[k]: m for k, m in fitted.items()}
    eval_models["Logistic Regression (tanpa normalisasi)"] = naive_lr

    rows, examples = [], []
    for pname in ["asli"] + list(PERTURBATIONS):
        texts = spam_texts if pname == "asli" else perturb_all(spam_texts, pname, RANDOM_SEED)
        for mname, model in eval_models.items():
            recall = float((model.predict(texts) == POS_LABEL).mean())
            rows.append({"model": mname, "perturbasi": pname,
                         "recall_spam": round(recall, 4), "n": len(texts)})
        if pname != "asli":
            for orig, pert in list(zip(spam_texts, texts))[:3]:
                examples.append({"perturbasi": pname, "asli": orig, "disamarkan": pert})
    pd.DataFrame(rows).to_csv(ARTIFACTS / "robustness.csv", index=False)
    pd.DataFrame(examples).to_csv(ARTIFACTS / "robustness_examples.csv", index=False)
    print("\nSelesai. Artefak tersimpan di", ARTIFACTS)


if __name__ == "__main__":
    main()
