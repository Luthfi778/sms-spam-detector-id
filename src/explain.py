"""Penjelasan prediksi: kata mana yang paling mendorong ke Spam atau Normal.

Hanya untuk model linear (Logistic Regression): kontribusi = bobot fitur x nilai TF-IDF.
"""
from .config import POS_LABEL


def explain_linear(pipeline, text: str, top_k: int = 8):
    """Kembalikan (kata_pendorong_spam, kata_pendorong_normal) sebagai list (kata, skor)."""
    feats = pipeline.named_steps["features"]
    clf = pipeline.named_steps["clf"]
    sign = 1.0 if list(clf.classes_).index(POS_LABEL) == 1 else -1.0

    x = feats.transform([text])
    names = feats.get_feature_names_out()
    coef = clf.coef_[0] * sign
    contrib = []
    for idx, val in zip(x.indices, x.data):
        name = names[idx]
        if name.startswith("word__"):          # tampilkan kata, bukan n-gram karakter
            contrib.append((name[len("word__"):], float(val * coef[idx])))

    to_spam = sorted([c for c in contrib if c[1] > 0], key=lambda c: -c[1])[:top_k]
    to_normal = sorted([c for c in contrib if c[1] < 0], key=lambda c: c[1])[:top_k]
    return to_spam, to_normal
