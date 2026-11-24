"""Halaman tentang data: sumber, distribusi, duplikat, kata teratas."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.app_utils import (FOOTER, WIDTH_CHART, load_dataset_clean, load_summary,
                           require_artifacts, show_fig, themed)
from src.config import LABELS

st.set_page_config(page_title="Tentang Data", page_icon="🗂️", layout="wide")
require_artifacts()

s = load_summary()
d = s["dedup"]
df = load_dataset_clean()

st.title("🗂️ Tentang Data")
st.markdown(
    "**Sumber:** *Indonesian SMS Spam Detection* (Kaggle), berkas `indo_spam.csv` yang "
    "juga tersedia di repositori GitHub `gevabriel/dataset`. Kolom asli: `Kategori` "
    "(ham/Spam) dan `Pesan`.\n\n"
    "**Catatan penting:** isi pesan (mis. nomor shortcode, \"ringtone\", mata uang pound) "
    "menunjukkan bahwa dataset ini tampaknya terjemahan dari dataset berbahasa Inggris "
    "(UCI SMS Spam Collection), bukan SMS asli Indonesia. Artinya model lebih mengenali "
    "pola spam gaya Inggris yang diterjemahkan daripada modus penipuan lokal seperti "
    "pinjol ilegal atau kurir palsu. Ini keterbatasan utama proyek."
)

# ---- Pembersihan ------------------------------------------------------------
st.subheader("Pembersihan data")
c = st.columns(4)
c[0].metric("Data mentah", d["n_raw"])
c[1].metric("Duplikat dibuang", d["n_duplicates"])
c[2].metric("Data bersih", d["n_clean"])
c[3].metric("Train / Test", f"{s['n_train']} / {s['n_test']}")
st.caption("Duplikat (termasuk yang sama setelah preprocessing) dibuang **sebelum** "
           "pembagian data, agar pesan yang sama tidak muncul di data latih dan data uji "
           "(kebocoran data yang membuat skor tampak terlalu bagus).")

# ---- Distribusi kelas -------------------------------------------------------
st.subheader("Distribusi kelas")
dist = pd.DataFrame({
    "Mentah": s["counts_raw"], "Bersih": s["counts_clean"],
    "Train": s["counts_train"], "Test": s["counts_test"],
}).reindex(LABELS)
left, right = st.columns([1, 1])
left.dataframe(dist, width="stretch")
share = (dist["Bersih"] / dist["Bersih"].sum()).round(3)
left.caption(f"Proporsi Spam pada data bersih: **{share['Spam']:.1%}**. Kelas tidak "
             "seimbang secara moderat, sehingga model memakai pembobotan kelas dan "
             "evaluasi berfokus pada precision/recall/F1 kelas Spam, bukan accuracy saja.")
right.bar_chart(dist[["Train", "Test"]])

# ---- Panjang pesan ----------------------------------------------------------
st.subheader("Panjang pesan (karakter)")
df["panjang"] = df["text"].str.len()
clip = float(df["panjang"].quantile(0.99))      # potong ekor panjang agar grafik terbaca
with themed():
    fig, ax = plt.subplots(figsize=(6, 3))
    for lab in LABELS:
        ax.hist(df.loc[df["label"] == lab, "panjang"].clip(upper=clip), bins=30,
                range=(0, clip), alpha=0.7, label=lab)
    ax.set_xlim(0, clip)
    ax.set_xlabel("Jumlah karakter")
    ax.set_ylabel("Jumlah pesan")
    ax.legend()
    fig.tight_layout()
show_fig(fig, WIDTH_CHART)
st.caption(f"Sumbu dipotong pada persentil ke-99 ({clip:.0f} karakter); pesan yang lebih panjang "
           "dihitung di batang terakhir. Rata-rata panjang — " + " · ".join(
    f"{lab}: {df.loc[df['label'] == lab, 'panjang'].mean():.0f}" for lab in LABELS))

# ---- Kata teratas -----------------------------------------------------------
st.subheader("Kata paling sering muncul per kelas (data latih)")
cols = st.columns(2)
for col, lab in zip(cols, LABELS):
    col.write(f"**{lab}**")
    col.dataframe(pd.DataFrame(s["top_words"][lab], columns=["Kata", "Jumlah pesan"]),
                  hide_index=True, width="stretch")

# ---- Contoh data ------------------------------------------------------------
st.subheader("Contoh data")
lab = st.radio("Kelas", LABELS, horizontal=True, index=1)
st.dataframe(df[df["label"] == lab][["text"]].head(15).rename(columns={"text": "Pesan"}),
             width="stretch", hide_index=True)

st.divider()
st.caption(FOOTER)
