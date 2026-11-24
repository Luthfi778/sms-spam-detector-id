"""Halaman analisis kesalahan (FP/FN) dan uji ketahanan terhadap pesan yang disamarkan."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.app_utils import (FOOTER, load_metrics, load_predictions, load_robustness,
                           require_artifacts)
from src.config import POS_LABEL

st.set_page_config(page_title="Analisis Kesalahan", page_icon="🔍", layout="wide")
require_artifacts()

metrics = load_metrics()
models = metrics["models"]
preds = load_predictions()

st.title("🔍 Analisis Kesalahan")

# ---- FP / FN per model ------------------------------------------------------
st.subheader("False positive & false negative")
st.write("- **False positive (FP)**: pesan *Normal* yang salah ditandai *Spam* "
         "(pengguna bisa kehilangan pesan penting).\n"
         "- **False negative (FN)**: pesan *Spam* yang lolos dianggap *Normal* "
         "(pengguna tetap terpapar penipuan).")

key = st.selectbox("Model", list(models), format_func=lambda k: models[k]["name"])
pred_col, proba_col = f"pred_{key}", f"proba_spam_{key}"
wrong = preds[preds["label"] != preds[pred_col]]
fp = wrong[(wrong["label"] != POS_LABEL)]
fn = wrong[(wrong["label"] == POS_LABEL)]

c1, c2, c3 = st.columns(3)
c1.metric("Total salah", len(wrong), f"dari {len(preds)} pesan uji", delta_color="off")
c2.metric("False positive", len(fp))
c3.metric("False negative", len(fn))

tab_fp, tab_fn = st.tabs([f"False positive ({len(fp)})", f"False negative ({len(fn)})"])
cols = {"text": "Pesan", proba_col: "P(Spam)"}
with tab_fp:
    st.dataframe(fp[list(cols)].rename(columns=cols).sort_values("P(Spam)", ascending=False),
                 width="stretch", hide_index=True)
with tab_fn:
    st.dataframe(fn[list(cols)].rename(columns=cols).sort_values("P(Spam)"),
                 width="stretch", hide_index=True)

# ---- Kesalahan bersama ------------------------------------------------------
st.subheader("Pesan yang salah diprediksi oleh semua model")
pred_cols = [f"pred_{k}" for k in models if f"pred_{k}" in preds]
all_wrong = preds[(preds[pred_cols].ne(preds["label"], axis=0)).all(axis=1)]
st.write(f"**{len(all_wrong)}** pesan salah di semua model. Ini kandidat kuat untuk "
         "label yang meragukan atau pesan yang memang ambigu; baca satu per satu untuk "
         "mencari pola.")
st.dataframe(all_wrong[["text", "label"]].rename(columns={"text": "Pesan", "label": "Label asli"}),
             width="stretch", hide_index=True)

# ---- Uji ketahanan ----------------------------------------------------------
st.subheader("Uji ketahanan terhadap pesan yang disamarkan")
st.write("Pesan **Spam** di data uji diubah seperti trik penipu menghindari filter "
         "(huruf dipisah spasi/titik, angka pengganti huruf, typo), lalu diukur "
         "**recall Spam**-nya. Model *Logistic Regression (tanpa normalisasi)* adalah "
         "pembanding yang tidak memakai langkah normalisasi penyamaran.")
rob, examples = load_robustness()
pivot = rob.pivot(index="model", columns="perturbasi", values="recall_spam")
pivot = pivot[["asli"] + [c for c in pivot.columns if c != "asli"]]


def band(v):
    """Pita warna diskret; latar gelap + teks putih terbaca di tema terang maupun gelap."""
    if v >= 0.93:
        bg = "#1f7a3d"      # hijau
    elif v >= 0.90:
        bg = "#a8650d"      # oranye
    else:
        bg = "#b3261e"      # merah
    return f"background-color:{bg};color:#ffffff"


styler = pivot.style.format("{:.3f}")
styler = (styler.map if hasattr(styler, "map") else styler.applymap)(band)  # pandas lama/baru
st.dataframe(styler, width="stretch")
st.caption("Warna: hijau ≥ 0,93 · oranye 0,90–0,93 · merah < 0,90 (recall Spam).")
n = int(rob["n"].iloc[0])
st.caption(f"Jumlah pesan Spam uji: {n}. Dengan sampel sekecil ini, selisih 1 pesan "
           f"≈ {1 / n:.1%} recall, jadi selisih kecil jangan ditafsirkan berlebihan. "
           "Ini simulasi sederhana, bukan penyerang yang beradaptasi terhadap model.")
with st.expander("Contoh pesan sebelum dan sesudah disamarkan"):
    st.dataframe(examples.rename(columns={"perturbasi": "Perturbasi", "asli": "Asli",
                                          "disamarkan": "Disamarkan"}),
                 width="stretch", hide_index=True)

st.divider()
st.caption(FOOTER)
