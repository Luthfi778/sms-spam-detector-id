"""Preprocessing teks SMS berbahasa Indonesia.

Fungsi `clean_text` dipakai OLEH TRAINING DAN APLIKASI (lewat pipeline yang sama),
sehingga hasil di web identik dengan hasil evaluasi.

Prinsip: URL, nominal uang, dan angka TIDAK dibuang, tetapi diganti token
karena itu sinyal kuat penipuan/spam. Teks yang sengaja disamarkan penipu
(mis. "G R A T I S", "g.r.a.t.i.s", "h4d14h") dinormalkan kembali.
"""
import re

URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|co\.id|net|org|id|info|biz|xyz|top)\b\S*)",
    re.IGNORECASE,
)
MONEY_RE = re.compile(r"(\b(?:rp|idr)\.?\s*\d[\d.,]*|[£$€]\s*\d[\d.,]*)", re.IGNORECASE)
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"}

# Kamus normalisasi bahasa gaul/singkatan yang umum di SMS (sengaja dibuat kecil).
SLANG = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak", "tdk": "tidak",
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "utk": "untuk", "krn": "karena",
    "jgn": "jangan", "hub": "hubungi", "kntr": "kantor", "dpt": "dapat",
    "mendpt": "mendapat", "td": "tadi", "sdh": "sudah", "udh": "sudah",
    "trm": "terima", "tlp": "telepon", "pkt": "paket", "bs": "bisa", "tgl": "tanggal",
}

# Stopword ringan, hanya untuk ringkasan "kata teratas" di halaman data.
STOPWORDS = set(
    "yang dan di ke dari untuk dengan ini itu saya anda kamu aku kita ada tidak akan "
    "atau pada juga sudah bisa lebih dalam adalah oleh karena jika kalau mau apa "
    "sebagai saat telah lagi ya sih dong nya angka nomorpanjang urltoken uang".split()
)


def _deobfuscate(tokens):
    """Gabungkan deretan karakter tunggal ("g r a t i s" -> "gratis")."""
    out, i = [], 0
    while i < len(tokens):
        j = i
        while j < len(tokens) and len(tokens[j]) == 1 and (
            tokens[j].isalpha() or tokens[j] in LEET
        ):
            j += 1
        run = tokens[i:j]
        if len(run) >= 3 and any(t.isalpha() for t in run):
            out.append("".join(LEET.get(t, t) for t in run))
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return out


def _normalize_token(tok):
    letters = sum(c.isalpha() for c in tok)
    digits = sum(c.isdigit() for c in tok)
    # Token campuran huruf+angka ("h4d14h") dianggap leetspeak.
    if letters >= 2 and digits and digits <= len(tok) / 2:
        tok = "".join(LEET.get(c, c) for c in tok)
    elif tok.isdigit():
        return "nomorpanjang" if len(tok) >= 5 else "angka"
    return SLANG.get(tok, tok)


def clean_text(text) -> str:
    """Bersihkan satu pesan. Selalu mengembalikan string huruf kecil."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_RE.sub(" urltoken ", text)
    text = MONEY_RE.sub(" uang ", text)
    tokens = [t for t in NON_ALNUM_RE.split(text) if t]
    tokens = _deobfuscate(tokens)
    return " ".join(_normalize_token(t) for t in tokens)


def naive_clean(text) -> str:
    """Pembersihan naif (hanya huruf kecil + buang simbol), TANPA normalisasi penyamaran.

    Dipakai hanya sebagai pembanding (ablation) pada uji ketahanan.
    """
    if not isinstance(text, str):
        return ""
    return " ".join(t for t in NON_ALNUM_RE.split(text.lower()) if t)
