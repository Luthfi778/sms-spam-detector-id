"""Halaman evaluasi: metrik, confusion matrix, kurva PR, waktu inferensi."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_recall_curve

from src.app_utils import (FOOTER, WIDTH_CHART, WIDTH_CM, load_metrics, load_predictions,
                           plot_confusion_matrix, require_artifacts, show_fig, themed)
from src.theme import apply_theme, scene
from src.config import POS_LABEL

st.set_page_config(page_title="Evaluasi Model", page_icon="📊", layout="wide")
apply_theme()
require_artifacts()

metrics = load_metrics()
meta, models = metrics["meta"], metrics["models"]

st.title("📊 Evaluasi Model")
st.write(f"Semua metrik dihitung pada **data uji** ({meta['n_test']} pesan) yang tidak "
         f"dipakai saat training. Kelas positif = **{POS_LABEL}**. "
         f"Split stratified, seed {meta['seed']}.")

# ---- Tabel perbandingan -----------------------------------------------------
rows = []
for k, m in models.items():
    rows.append({
        "Model": m["name"],
        "Accuracy": m["accuracy"],
        "Precision (Spam)": m["precision_spam"],
        "Recall (Spam)": m["recall_spam"],
        "F1 (Spam)": m["f1_spam"],
        "F1 macro": m["f1_macro"],
        "PR-AUC": m["pr_auc"],
        "ROC-AUC": m["roc_auc"],
        "F1 macro CV (train)": m.get("cv_f1_macro"),
        "Waktu latih (dtk)": m.get("train_seconds"),
        "Inferensi (ms/pesan)": m.get("infer_ms"),
    })
table = pd.DataFrame(rows).set_index("Model")


def highlight_best(col):
    """Sorot nilai tertinggi. Latar hijau tua + teks putih: terbaca di tema terang dan gelap."""
    style = "background-color:#1f7a3d;color:#ffffff;font-weight:700"
    return [style if v == col.max() else "" for v in col]


BEST_COLS = ["Accuracy", "Precision (Spam)", "Recall (Spam)", "F1 (Spam)", "F1 macro",
             "PR-AUC", "ROC-AUC"]
fmt = {c: "{:.3f}" for c in table.columns if c not in ("Waktu latih (dtk)", "Inferensi (ms/pesan)")}
fmt.update({"Waktu latih (dtk)": "{:.1f}", "Inferensi (ms/pesan)": "{:.2f}"})
st.subheader("Perbandingan model")
st.dataframe(table.style.format(fmt, na_rep="-").apply(highlight_best, subset=BEST_COLS),
             width="stretch")
st.caption("Sel hijau = nilai tertinggi per kolom (jika sama, keduanya disorot). 'F1 macro CV' adalah skor validasi "
           "silang 5-lipat pada data latih yang dipakai untuk memilih hyperparameter.")

# ---- Bar chart --------------------------------------------------------------
st.subheader("Precision, Recall, dan F1 kelas Spam")
st.bar_chart(table[["Precision (Spam)", "Recall (Spam)", "F1 (Spam)"]], height=320)

# ---- Detail per model -------------------------------------------------------
st.subheader("Detail per model")
tabs = st.tabs([m["name"] for m in models.values()])
for tab, (k, m) in zip(tabs, models.items()):
    with tab:
        c1, c2 = st.columns([1, 1.4])
        with c1:
            show_fig(plot_confusion_matrix(m["confusion_matrix"], m["name"]), WIDTH_CM)
            tn, fp = m["confusion_matrix"][0]
            fn, tp = m["confusion_matrix"][1]
            st.caption(f"TN={tn} · **FP={fp}** (Normal salah jadi Spam) · "
                       f"**FN={fn}** (Spam lolos) · TP={tp}")
        with c2:
            pc = pd.DataFrame(m["per_class"]).T
            pc.index.name = "Kelas"
            st.dataframe(pc.style.format(
                {"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}", "support": "{:.0f}"}),
                width="stretch")
            if m.get("best_params"):
                st.write("Hyperparameter terpilih:", m["best_params"])

# ---- Kurva Precision-Recall -------------------------------------------------
st.subheader("Kurva Precision–Recall")
preds = load_predictions()
y = (preds["label"] == POS_LABEL).to_numpy()
with themed():
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, m in models.items():
        col = f"proba_spam_{k}"
        if col in preds:
            p, r, _ = precision_recall_curve(y, preds[col])
            ax.plot(r, p, linewidth=2, label=f"{m['name']} (AP={m['pr_auc']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.5, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
show_fig(fig, WIDTH_CHART)
st.caption("Kurva yang lebih mendekati pojok kanan atas lebih baik. Menggeser ambang "
           "keputusan di halaman Deteksi bergerak sepanjang kurva ini.")

st.divider()
st.caption(FOOTER)

scene()
