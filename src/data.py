"""Memuat, membersihkan duplikat, dan membagi dataset."""
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, RANDOM_SEED, TEST_SIZE
from .preprocessing import clean_text


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    """Kolom asli: Kategori (ham/Spam), Pesan. Diubah menjadi label (Normal/Spam), text."""
    df = pd.read_csv(path).rename(columns={"Kategori": "label", "Pesan": "text"})
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(
        {"ham": "Normal", "spam": "Spam"}
    )
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    return df.reset_index(drop=True)


def deduplicate(df: pd.DataFrame):
    """Buang pesan kosong, duplikat (setelah dibersihkan), dan duplikat berlabel ganda.

    Dilakukan SEBELUM split supaya pesan yang sama tidak muncul di train dan test
    (kebocoran data yang membuat skor tampak terlalu bagus).
    """
    df = df.copy()
    df["_key"] = df["text"].map(clean_text)
    info = {"n_raw": len(df)}

    empty = df["_key"] == ""
    info["n_empty"] = int(empty.sum())
    df = df[~empty]

    n_labels = df.groupby("_key")["label"].nunique()
    conflict_keys = n_labels[n_labels > 1].index
    info["n_conflicting"] = int(df["_key"].isin(conflict_keys).sum())
    df = df[~df["_key"].isin(conflict_keys)]

    before = len(df)
    df = df.drop_duplicates(subset="_key", keep="first")
    info["n_duplicates"] = before - len(df)
    info["n_clean"] = len(df)
    return df.drop(columns="_key").reset_index(drop=True), info


def split(df: pd.DataFrame):
    """Split stratified 80/20 dengan seed tetap (dipakai juga oleh skrip IndoBERT)."""
    train, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_SEED
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
