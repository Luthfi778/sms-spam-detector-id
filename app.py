"""Halaman utama: deteksi SMS spam/penipuan berbahasa Indonesia.

Jalankan:  streamlit run app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from src.app_utils import (FOOTER, available_models, load_metrics, load_model,
                           require_artifacts)
from src.config import EXPLAIN_MODEL, POS_LABEL
from src.explain import explain_linear
from src.preprocessing import clean_text

st.set_page_config(page_title="Detektor SMS Spam", page_icon="🛡️", layout="wide")
require_artifacts()

SAMPLES = {
    "Hadiah undian": "Selamat! Nomor Anda memenangkan hadiah Rp 50.000.000 dari undian "
                     "resmi. Hubungi 08123456789 sekarang untuk klaim hadiah GRATIS Anda.",
    "Pulsa gratis": "GRATIS! Kirim TXT MENANG ke 87121 untuk mendapatkan ringtone dan "
                    "pulsa gratis. Hanya 3000 per minggu. Berlangganan sekarang!",
    "Samaran huruf": "S e l a m a t, anda terpilih mendapat h4d14h g.r.a.t.i.s! "
                     "Klik http://bit.ly/klaim-hadiah sekarang juga.",
    "Pesan biasa": "Assalamualaikum Pak, tadi saya meninggalkan proposal di meja Bapak "
                   "di atas printer. Terima kasih Pak.",
    "Ajakan teman": "Nanti malam jadi makan di tempat biasa? Aku sampai sekitar jam tujuh.",
}

metrics = load_metrics()
models = available_models(metrics)

# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Pengaturan")
    default = list(models).index("lr") if "lr" in models else 0
    key = st.selectbox("Model", list(models), index=default,
                       format_func=lambda k: models[k])
    threshold = st.slider(
        "Ambang Spam", 0.05, 0.95, 0.50, 0.05,
        help="Pesan dinilai Spam jika probabilitasnya ≥ ambang. Ambang lebih rendah "
             "→ lebih banyak spam tertangkap (recall naik) tetapi pesan normal lebih "
             "sering salah ditandai (precision turun).")
    m = metrics["models"][key]
    st.caption(f"F1 Spam pada data uji: **{m['f1_spam']:.3f}** · "
               f"Recall: **{m['recall_spam']:.3f}** · Precision: **{m['precision_spam']:.3f}**")
    st.caption("Pesan yang Anda ketik **tidak disimpan**.")

# ---- Konten ----------------------------------------------------------------
st.title("🛡️ Detektor SMS Spam & Penipuan")
st.write("Tempel pesan SMS/WhatsApp berbahasa Indonesia, lalu sistem memperkirakan "
         "apakah pesan itu **Spam** atau **Normal**.")

if "text" not in st.session_state:
    st.session_state["text"] = ""

st.write("**Contoh pesan** (klik untuk mengisi):")
cols = st.columns(len(SAMPLES))
for col, (name, msg) in zip(cols, SAMPLES.items()):
    if col.button(name, width="stretch"):
        st.session_state["text"] = msg

text = st.text_area("Isi pesan", key="text", height=140,
                    placeholder="Tempel pesan di sini…")

if st.button("Periksa", type="primary", disabled=not text.strip()):
    model = load_model(key)
    classes = list(model.classes_)
    p_spam = float(model.predict_proba([text])[0][classes.index(POS_LABEL)])
    is_spam = p_spam >= threshold

    left, right = st.columns([1, 1])
    with left:
        if is_spam:
            st.error(f"### ⚠️ {POS_LABEL}")
        else:
            st.success("### ✅ Normal")
        st.metric("Probabilitas Spam", f"{p_spam:.1%}")
        st.progress(min(max(p_spam, 0.0), 1.0))
        st.caption(f"Model: {models[key]} · ambang {threshold:.2f}")
        with st.expander("Teks setelah preprocessing"):
            st.code(clean_text(text) or "(kosong)", language=None)

    with right:
        st.write("**Kata yang paling memengaruhi keputusan**")
        if EXPLAIN_MODEL in models:
            to_spam, to_normal = explain_linear(load_model(EXPLAIN_MODEL), text)
            rows = ([{"kata": w, "pengaruh": s, "arah": "→ Spam"} for w, s in to_spam] +
                    [{"kata": w, "pengaruh": s, "arah": "→ Normal"} for w, s in to_normal])
            if rows:
                df = pd.DataFrame(rows).set_index("kata")
                st.bar_chart(df["pengaruh"], horizontal=True)
                st.caption("Dihitung dari bobot Logistic Regression (positif = condong "
                           "ke Spam, negatif = condong ke Normal).")
            else:
                st.info("Tidak ada kata yang dikenali model pada pesan ini.")

st.divider()
st.caption(FOOTER)
