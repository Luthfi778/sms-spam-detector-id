"""Uji ketahanan sederhana: pesan Spam yang sengaja disamarkan penipu.

Ini simulasi serangan evasi yang sederhana (bukan penyerang adaptif): kita ubah
pesan Spam di data uji, lalu ukur seberapa turun recall tiap model.
"""
import re

import numpy as np

LEET_MAP = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}
WORD_RE = re.compile(r"[A-Za-z]+")


def _per_word(text, fn, rng, min_len, prob):
    def repl(m):
        w = m.group(0)
        return fn(w, rng) if len(w) >= min_len and rng.random() < prob else w
    return WORD_RE.sub(repl, text)


def spasi_antar_huruf(text, rng):
    return _per_word(text, lambda w, r: " ".join(w), rng, 5, 0.5)


def titik_antar_huruf(text, rng):
    return _per_word(text, lambda w, r: ".".join(w), rng, 5, 0.5)


def leetspeak(text, rng):
    return _per_word(
        text, lambda w, r: "".join(LEET_MAP.get(c.lower(), c) for c in w), rng, 4, 0.5
    )


def hapus_huruf(text, rng):
    """Typo: hapus satu huruf di tengah kata (tidak dinormalkan oleh preprocessing)."""
    def fn(w, r):
        i = int(r.integers(1, len(w) - 1))
        return w[:i] + w[i + 1:]
    return _per_word(text, fn, rng, 5, 0.3)


def kapital_semua(text, rng):
    return text.upper()


PERTURBATIONS = {
    "kapital_semua": kapital_semua,
    "spasi_antar_huruf": spasi_antar_huruf,
    "titik_antar_huruf": titik_antar_huruf,
    "leetspeak": leetspeak,
    "hapus_huruf (typo)": hapus_huruf,
}


def perturb_all(texts, name, seed=42):
    rng = np.random.default_rng(seed)
    fn = PERTURBATIONS[name]
    return [fn(t, rng) for t in texts]
