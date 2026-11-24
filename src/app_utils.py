"""Helper bersama untuk semua halaman Streamlit (memuat artefak hasil training)."""
import json
import sys
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:          # supaya `import src...` jalan dari halaman manapun
    sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS, LABELS, MODELS_DIR  # noqa: E402

matplotlib.rcParams.update({"figure.dpi": 100})

FOOTER = (
    "Model dilatih pada dataset SMS berbahasa Indonesia berukuran kecil (±1.900 pesan "
    "setelah pembersihan). Hasilnya bisa salah, terutama pada pola penipuan baru. "
    "Gunakan sebagai alat bantu, bukan satu-satunya dasar keputusan."
)

# Lebar tampilan gambar (piksel) agar tidak melebar memenuhi layar.
WIDTH_CM = 340      # confusion matrix
WIDTH_CHART = 560   # kurva PR dan histogram


# ---------------------------------------------------------------- tema & gambar
def is_dark() -> bool:
    """True jika tema Streamlit saat ini gelap."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return st.get_option("theme.base") == "dark"


def _theme_rc() -> dict:
    if not is_dark():
        return {}
    bg, fg = "#0e1117", "#fafafa"
    return {
        "figure.facecolor": bg, "axes.facecolor": bg, "savefig.facecolor": bg,
        "text.color": fg, "axes.labelcolor": fg, "axes.edgecolor": "#8a8d93",
        "xtick.color": fg, "ytick.color": fg, "grid.color": "#4a4d55",
        "legend.facecolor": "#1a1d24", "legend.edgecolor": "#4a4d55",
    }


@contextmanager
def themed():
    """Buat figur di dalam blok ini agar warnanya mengikuti tema terang/gelap."""
    with plt.rc_context(_theme_rc()):
        yield


def show_fig(fig, width: int = WIDTH_CHART):
    """Tampilkan figur matplotlib dengan lebar tetap (piksel), bukan selebar layar."""
    buf = BytesIO()
    with plt.rc_context(_theme_rc()):
        fig.savefig(buf, format="png", dpi=170, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    st.image(buf.getvalue(), width=width)


def plot_confusion_matrix(cm, title: str):
    cm = np.array(cm)
    with themed():
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(2), LABELS)
        ax.set_yticks(range(2), LABELS)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Label asli")
        ax.set_title(title, fontsize=10)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=12,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------- artefak
def _trained_sklearn_version():
    p = ARTIFACTS / "versions.json"
    try:
        return json.loads(p.read_text()).get("scikit-learn")
    except Exception:
        return None


def require_artifacts():
    """Pastikan artefak ada DAN dilatih dengan versi scikit-learn yang sama.

    Model .joblib tidak kompatibel antar versi scikit-learn (contoh gejalanya:
    "AttributeError: 'LogisticRegression' object has no attribute 'multi_class'").
    Jika berbeda, tawarkan latih ulang (hanya sekitar satu menit).
    """
    has_metrics = (ARTIFACTS / "metrics.json").exists()
    trained = _trained_sklearn_version()
    if has_metrics and trained == sklearn.__version__:
        return

    if not has_metrics:
        st.error("Artefak model belum ada. Latih dulu dengan tombol di bawah, "
                 "atau jalankan `python -m src.train` di terminal.")
    else:
        st.warning(
            f"Model di folder `artifacts/` dilatih dengan scikit-learn "
            f"**{trained or 'versi tidak diketahui'}**, sedangkan di komputer ini terpasang "
            f"**{sklearn.__version__}**. Model perlu dilatih ulang agar kompatibel "
            "(sekitar 1 menit, hanya sekali).")
    if st.button("Latih ulang model sekarang", type="primary"):
        with st.spinner("Melatih model… mohon tunggu."):
            from src.train import main as train_main
            train_main()
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    st.caption("Alternatif: jalankan `python -m src.train` di terminal, lalu muat ulang halaman.")
    st.stop()


@st.cache_resource
def load_model(key: str):
    return joblib.load(MODELS_DIR / f"{key}.joblib")


@st.cache_data
def load_metrics() -> dict:
    return json.loads((ARTIFACTS / "metrics.json").read_text())


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "test_predictions.csv")


@st.cache_data
def load_summary() -> dict:
    return json.loads((ARTIFACTS / "data_summary.json").read_text())


@st.cache_data
def load_dataset_clean() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "dataset_clean.csv")


@st.cache_data
def load_robustness():
    return (pd.read_csv(ARTIFACTS / "robustness.csv"),
            pd.read_csv(ARTIFACTS / "robustness_examples.csv"))


def available_models(metrics: dict) -> dict:
    """key -> nama tampilan, hanya model yang punya file .joblib."""
    return {k: v["name"] for k, v in metrics["models"].items()
            if (MODELS_DIR / f"{k}.joblib").exists()}
