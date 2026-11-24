import pytest

from src.data import deduplicate, load_dataset, split
from src.explain import explain_linear
from src.preprocessing import clean_text, naive_clean


@pytest.mark.parametrize("raw, expected", [
    ("G R A T I S ringtone!", "gratis ringtone"),
    ("g.r.a.t.i.s hadiah", "gratis hadiah"),
    ("Dapat h4d14h sekarang", "dapat hadiah sekarang"),
    ("Klik http://bit.ly/abc", "klik urltoken"),
    ("Menang Rp. 75 juta", "menang uang juta"),
    ("Hub 08123456789", "hubungi nomorpanjang"),
    ("Maaf td ninggalin proposal", "maaf tadi ninggalin proposal"),
])
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_naive_clean_does_not_deobfuscate():
    assert naive_clean("G.R.A.T.I.S") == "g r a t i s"


def test_dedup_and_split_have_no_overlap():
    df, info = deduplicate(load_dataset())
    assert info["n_clean"] == len(df) <= info["n_raw"]
    train, test = split(df)
    assert set(train["text"]).isdisjoint(set(test["text"]))
    # proporsi kelas terjaga (stratified)
    assert abs((train["label"] == "Spam").mean() - (test["label"] == "Spam").mean()) < 0.02


def test_explain_linear_returns_spam_words():
    import joblib
    from src.config import MODELS_DIR
    model = joblib.load(MODELS_DIR / "lr.joblib")
    to_spam, _ = explain_linear(model, "menang hadiah gratis sekarang")
    assert to_spam and all(score > 0 for _, score in to_spam)
