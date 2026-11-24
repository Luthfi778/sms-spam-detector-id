"""Tema visual 'Derby': langit biru + awan, navigasi atas, kartu, rumput dan chibi.

Pakai di setiap halaman:
    apply_theme()            # tepat setelah st.set_page_config(...)  (halaman utama: home=True)
    ...konten...
    scene()                  # paling akhir: rumput di dasar halaman (chibi=True hanya di halaman utama)
Gambar ada di assets/, font opsional di static/fonts/ (lihat static/fonts/BACA-DULU.txt).
"""
import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent.parent / "assets"
BLUE, DARK, PINK, GREEN = "#1d63d8", "#1448a8", "#e6246e", "#14803c"

# Perpaduan font: Tactic Sans -> Arial Rounded -> rounded gratis (Nunito / M PLUS Rounded 1c) -> Arial.
# Tactic Sans & Arial Rounded hanya tampil jika terpasang di komputer pengguna (atau file font
# diletakkan di static/fonts/); selain itu browser otomatis memakai font berikutnya.
F_BODY = ("'Tactic Sans','TacticSans','Arial Rounded MT','Arial Rounded MT Bold',"
          "'Nunito','M PLUS Rounded 1c',Arial,sans-serif")
F_HEAD = ("'Tactic Sans','TacticSans','Arial Rounded MT Bold','Arial Rounded MT',"
          "'M PLUS Rounded 1c','Nunito',Arial,sans-serif")


@lru_cache(maxsize=None)
def img(name: str) -> str:
    """Data URI untuk gambar di assets/ (di-cache agar tidak dibaca ulang)."""
    data = base64.b64encode((ASSETS / name).read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def _blades(heights, shift=0.0) -> str:
    """clip-path polygon bilah rumput; heights = tinggi puncak (%) yang diulang."""
    n = 64
    pts = ["0% 100%", "0% 62%"]
    for i in range(n):
        x0 = (i + shift) / n * 100
        peak = heights[i % len(heights)]
        pts += [f"{x0:.2f}% 62%", f"{x0 + 0.5 / n * 100:.2f}% {peak}%"]
    pts += ["100% 62%", "100% 100%"]
    return "polygon(" + ",".join(pts) + ")"


def _clouds() -> str:
    """Awan dari lingkaran putih bertumpuk (posisi % + offset px, ukuran skala)."""
    spots = [(10, 12, 1.0), (74, 8, 1.3), (42, 30, 0.8), (92, 40, 1.0),
             (6, 58, 1.1), (58, 68, 0.9), (24, 88, 1.2), (86, 92, 0.8)]
    parts = [(0, 0, 40), (-44, 12, 30), (46, 10, 32), (86, 18, 22), (-82, 20, 20)]
    layers = []
    for cx, cy, s in spots:
        for dx, dy, r in parts:
            r2 = r * s
            layers.append(
                f"radial-gradient(circle {r2:.0f}px at calc({cx}% + {dx * s:.0f}px) "
                f"calc({cy}% + {dy * s:.0f}px),rgba(255,255,255,.96) {r2 - 3:.0f}px,"
                f"rgba(255,255,255,0) {r2:.0f}px)")
        layers.append(
            f"radial-gradient(ellipse {110 * s:.0f}px {24 * s:.0f}px at calc({cx}% + {20 * s:.0f}px) "
            f"calc({cy}% + {32 * s:.0f}px),rgba(255,255,255,.96) 70%,rgba(255,255,255,0) 100%)")
    return ",".join(layers)


def _css() -> str:
    back = _blades([8, 30, 0, 22, 12, 34])
    front = _blades([26, 6, 32, 14, 28, 4], shift=0.5)
    clouds = _clouds()
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@600;700;800;900&family=M+PLUS+Rounded+1c:wght@500;700;800;900&display=swap');
@font-face{{font-family:'Tactic Sans';font-weight:100 600;font-style:normal;
  src:local('Tactic Sans'),local('TacticSans'),local('TacticSans-Regular'),
      url('/app/static/fonts/TacticSans-Regular.woff2') format('woff2'),
      url('/app/static/fonts/TacticSans-Regular.otf') format('opentype');}}
@font-face{{font-family:'Tactic Sans';font-weight:700 900;font-style:normal;
  src:local('Tactic Sans Bold'),local('TacticSans-Bold'),
      url('/app/static/fonts/TacticSans-Bold.woff2') format('woff2'),
      url('/app/static/fonts/TacticSans-Bold.otf') format('opentype');}}

.stApp {{--f-body:{F_BODY};--f-head:{F_HEAD};}}
.stApp, .stApp p, .stApp label, .stApp li, .stApp button, .stApp textarea, .stApp input,
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stMetricLabel"] {{font-family:var(--f-body);}}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp [data-testid="stMetricValue"],
.stApp .d-title, .stApp .d-badge, .stApp .d-stamp, .stApp .d-rib, .stApp .d-marker {{font-family:var(--f-head);}}
.stApp span[data-testid="stIconMaterial"], .stApp [class*="material-symbols"] {{font-family:'Material Symbols Rounded' !important;}}

/* ---------- Langit + awan ---------- */
.stApp {{background:linear-gradient(180deg,#43b0ff 0%,#78cbff 38%,#b9e5ff 72%,#e6f6ff 100%) fixed !important;
  color:#12306b;overflow-x:hidden;}}
.stApp::before {{content:"";position:fixed;top:0;bottom:0;left:-12%;right:-12%;z-index:0;pointer-events:none;
  background:{clouds};animation:d-drift 90s ease-in-out infinite alternate;}}
@keyframes d-drift {{from {{transform:translateX(-3%);}} to {{transform:translateX(3%);}}}}
@media (prefers-reduced-motion:reduce) {{.stApp::before {{animation:none;}}}}
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] {{background:transparent !important;}}
[data-testid="stMain"] {{position:relative;z-index:1;overflow-x:hidden;}}
.block-container {{max-width:1100px;padding-top:1.4rem !important;padding-bottom:0 !important;}}

/* ---------- Navigasi atas ---------- */
.st-key-topnav {{background:rgba(255,255,255,.92);border:4px solid {BLUE};border-radius:999px;
  box-shadow:0 6px 0 {DARK};padding:6px 18px;margin:0 0 8px;}}
.st-key-topnav [data-testid="stHorizontalBlock"] {{align-items:center;gap:.5rem;}}
.st-key-topnav [data-testid="stPageLink"] a {{justify-content:center;border-radius:999px;padding:8px 10px;
  color:{DARK};font-weight:800;text-decoration:none;}}
.st-key-topnav [data-testid="stPageLink"] a p {{color:{DARK};font-weight:800;font-size:1rem;}}
.st-key-topnav [data-testid="stPageLink"] a:hover {{background:#e3f1ff;}}
.st-key-topnav [data-testid="stPageLink"] a[aria-current="page"] {{background:#ffd23f;box-shadow:0 3px 0 {DARK};}}
.d-navlogo img {{height:40px;width:auto;display:block;}}

/* ---------- Judul ---------- */
.stApp h1 {{color:#fff;font-weight:900;letter-spacing:.01em;
  text-shadow:0 4px 0 {DARK},3px 3px 0 {DARK},-3px 3px 0 {DARK},3px -2px 0 {DARK},-3px -2px 0 {DARK};}}
.stApp h2, .stApp h3 {{color:{DARK};font-weight:900;}}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{background:#e3f1ff;border-right:4px solid {BLUE};}}

/* ---------- Tombol ---------- */
.stApp .stButton > button {{background:{BLUE};border:3px solid #fff;border-radius:999px;
  box-shadow:0 4px 0 {DARK};min-height:46px;font-weight:800;transition:transform .08s;}}
.stApp .stButton > button p {{color:#fff;font-weight:800;}}
.stApp .stButton > button:hover {{transform:translateY(-2px);filter:brightness(1.08);}}
.stApp .stButton > button:active {{transform:translateY(3px);box-shadow:0 1px 0 {DARK};}}
.stApp .stButton > button[kind="primary"], .stApp .stButton > button[data-testid="stBaseButton-primary"] {{
  background:{PINK};min-height:58px;padding:0 44px;box-shadow:0 0 0 3px {BLUE},0 7px 0 #a3114d;}}
.stApp .stButton > button[kind="primary"] p {{font-size:1.4rem;}}
.stApp .stButton > button:disabled {{background:#9fb4d6;box-shadow:0 4px 0 #6f86ad;opacity:.85;}}
.st-key-card_input [data-testid="stColumn"]:nth-child(2) .stButton > button {{background:{PINK};box-shadow:0 4px 0 #a3114d;}}
.st-key-card_input [data-testid="stColumn"]:nth-child(3) .stButton > button {{background:{GREEN};box-shadow:0 4px 0 #0b5a28;}}
.st-key-card_input [data-testid="stColumn"]:nth-child(4) .stButton > button {{background:#b45309;box-shadow:0 4px 0 #7a3806;}}
.st-key-card_input [data-testid="stColumn"]:nth-child(5) .stButton > button {{background:#7c3aed;box-shadow:0 4px 0 #4c1d95;}}

/* ---------- Input & kartu ---------- */
div[data-baseweb="textarea"], div[data-baseweb="select"] > div {{border:3px solid #9cc4ff !important;
  border-radius:18px !important;background:#f4f9ff !important;}}
[data-testid="stTextArea"] textarea {{font-size:1.05rem;color:#12306b;}}
.st-key-card_input, .st-key-card_settings {{background:#fff;border:4px solid {BLUE};border-radius:28px;box-shadow:0 8px 0 {DARK};
  padding:26px 32px 30px;margin:8px 0 18px;}}
[data-testid="stMetric"] {{background:#fff;border:4px solid {BLUE};border-radius:22px;
  box-shadow:0 6px 0 {DARK};padding:14px 20px;}}
[data-testid="stMetricValue"] {{color:{BLUE};font-weight:900;}}
button[data-baseweb="tab"] p {{font-weight:800;}}
[data-testid="stDataFrame"] {{border:3px solid #9cc4ff;border-radius:16px;overflow:hidden;background:#fff;}}
[data-testid="stAlert"] {{border-radius:18px;}}
[data-testid="stExpander"] {{border:3px solid #9cc4ff;border-radius:18px;background:#fff;}}
[data-testid="stMain"] [data-testid="stCaptionContainer"] {{background:rgba(255,255,255,.88);
  border-radius:14px;padding:8px 14px;}}
[data-testid="stMain"] hr {{border-color:rgba(29,99,216,.35);}}

/* ---------- Hero ---------- */
.stApp .d-hero {{text-align:center;padding:34px 0 10px;width:100%;}}
.stApp .d-hero img {{height:76px;width:auto;}}
.stApp .d-badge {{display:inline-block;margin:8px 0 0;font-weight:900;font-size:1.05rem;color:{DARK};background:#ffd23f;
  border:3px solid #fff;border-radius:999px;padding:4px 20px;box-shadow:0 4px 0 {DARK};}}
.stApp .d-title {{font-weight:900;font-size:clamp(44px,7vw,92px);line-height:1;color:#fff;margin:16px 0 0;transform:rotate(-2deg);
  text-shadow:0 6px 0 {DARK},4px 4px 0 {DARK},-4px 4px 0 {DARK},4px -2px 0 {DARK},-4px -2px 0 {DARK};}}
.stApp .d-hero p.d-sub {{max-width:600px;margin:26px auto 22px !important;padding:12px 22px;text-align:center !important;
  font-size:1.1rem;font-weight:700;line-height:1.6;color:#12306b;background:rgba(255,255,255,.88);border-radius:20px;}}
.d-sub .s {{color:{PINK};}} .d-sub .n {{color:{GREEN};}}
.stApp .d-rib.g {{background:{GREEN};}}
.stApp .d-rib {{display:inline-block;font-weight:900;font-size:1.05rem;color:#fff;background:{BLUE};
  padding:7px 24px 7px 16px;border-radius:14px 40px 40px 14px;transform:rotate(-2deg);margin-bottom:6px;}}

/* ---------- Kartu hasil ---------- */
.d-result {{background:#fff;border:4px solid {BLUE};border-radius:28px;box-shadow:0 8px 0 {DARK};padding:24px 28px;
  display:flex;flex-wrap:wrap;gap:24px;align-items:center;margin:8px 0 18px;}}
.d-result img {{height:150px;width:auto;}}
.d-stamp {{font-weight:900;font-size:2.8rem;line-height:1;color:#fff;border:5px solid #fff;border-radius:20px;padding:14px 28px;}}
.d-stamp.spam {{background:{PINK};box-shadow:0 0 0 3px {PINK},0 8px 0 #a3114d;transform:rotate(-4deg);}}
.d-stamp.ok {{background:{GREEN};box-shadow:0 0 0 3px {GREEN},0 8px 0 #0b5a28;transform:rotate(3deg);}}
.d-lane {{flex:1 1 260px;min-width:240px;}}
.d-lane .row {{display:flex;justify-content:space-between;font-weight:800;font-size:.85rem;margin-bottom:8px;}}
.d-track {{position:relative;height:40px;border-radius:999px;border:4px solid {BLUE};
  background:repeating-linear-gradient(90deg,#c9a46a 0 40px,#d6b27a 40px 80px);}}
.d-flag {{position:absolute;right:0;top:0;bottom:0;width:20px;border-radius:0 999px 999px 0;
  background:repeating-conic-gradient(#12306b 0 25%,#fff 0 50%) 0 0/10px 10px;}}
.d-marker {{position:absolute;top:-12px;width:52px;height:52px;border-radius:50%;background:#ffd23f;border:4px solid #fff;
  box-shadow:0 4px 0 {DARK};display:flex;align-items:center;justify-content:center;font-weight:900;font-size:.85rem;color:{DARK};}}
.d-lane p {{margin:14px 0 0;font-weight:700;line-height:1.5;}}

/* ---------- Rumput di akhir halaman (ikut alur halaman, tidak menimpa teks) ---------- */
.d-scene {{position:relative;width:100vw;left:50%;transform:translateX(-50%);margin-top:64px;pointer-events:none;}}
.d-scene .b, .d-scene .f {{position:absolute;left:0;right:0;bottom:30px;height:62px;z-index:1;}}
.d-scene .b {{background:#2f9e44;clip-path:{back};}}
.d-scene .f {{background:#5fd068;clip-path:{front};}}
.d-scene .base {{z-index:1;position:absolute;left:0;right:0;bottom:0;height:42px;background:linear-gradient(#5fd068,#2f9e44);}}
.d-scene img {{position:absolute;bottom:44px;height:112px;width:auto;z-index:3;}}
@media (max-width:900px) {{.d-scene img {{display:none;}}}}
</style>
"""


def top_nav(home: bool) -> None:
    """Navigasi di bagian atas halaman (menggantikan navigasi sidebar bawaan)."""
    links = [("app.py", "Detektor"), ("pages/1_Evaluasi_Model.py", "Evaluasi Model"),
             ("pages/2_Analisis_Kesalahan.py", "Analisis Kesalahan"),
             ("pages/3_Tentang_Data.py", "Tentang Data")]
    with st.container(key="topnav"):
        cols = st.columns([1.1, 1, 1.3, 1.5, 1.2])
        if not home:
            cols[0].markdown(f'<div class="d-navlogo"><img src="{img("uma-logo.png")}" alt="Logo"></div>',
                             unsafe_allow_html=True)
        for col, (path, label) in zip(cols[1:], links):
            try:
                col.page_link(path, label=label)
            except Exception:   # mis. halaman dijalankan langsung, bukan lewat `streamlit run app.py`
                col.markdown(f"**{label}**")


def apply_theme(home: bool = False) -> None:
    """Pasang tema + navigasi atas. Panggil tepat setelah st.set_page_config(...)."""
    st.markdown(_css(), unsafe_allow_html=True)
    top_nav(home)


def scene(chibis: bool = True) -> None:
    """Rumput di dasar halaman. chibis=True (bawaan) menaruh lima chibi berdiri di depan rumput."""
    height = 156 if chibis else 96
    pics = ""
    if chibis:
        pos = [("rumput-chibi-0.png", 8), ("rumput-chibi-1.png", 29), ("rumput-chibi-2.png", 50),
               ("rumput-chibi-3.png", 71), ("rumput-chibi-4.png", 92)]
        pics = "".join(f'<img src="{img(n)}" alt="" style="left:{x}%">' for n, x in pos)
    st.markdown(f'<div class="d-scene" style="height:{height}px">{pics}'
                '<div class="b"></div><div class="f"></div><div class="base"></div></div>',
                unsafe_allow_html=True)


def hero(kicker: str = "★ LOMBA DIMULAI! ★") -> None:
    st.markdown(
        f'<div class="d-hero"><img src="{img("uma-logo.png")}" alt="Logo"><br>'
        f'<span class="d-badge">{kicker}</span>'
        '<div class="d-title">DETEKTOR<br>SMS SPAM</div>'
        '<p class="d-sub">Tempel pesan SMS/WhatsApp berbahasa Indonesia, lalu lihat apakah pesan itu '
        'berlari menuju garis <span class="s">Spam</span> atau <span class="n">Normal</span>!</p></div>',
        unsafe_allow_html=True)


def result_card(is_spam: bool, p_spam: float, threshold: float) -> None:
    """Kartu hasil: stempel SPAM!/NORMAL, lintasan dengan penanda probabilitas, dan chibi reaksi."""
    pct = min(max(p_spam, 0.0), 1.0) * 100
    pic = img("chibi-kaget.png" if is_spam else "chibi-wortel.png")
    stamp = ('<div class="d-stamp spam">SPAM!</div>' if is_spam
             else '<div class="d-stamp ok">NORMAL</div>')
    note = (f"Probabilitas spam {pct:.1f}% di atas ambang {threshold:.2f}, jadi pesan ini "
            "melewati garis finish sebagai <b>Spam</b>." if is_spam else
            f"Probabilitas spam {pct:.1f}% di bawah ambang {threshold:.2f}, jadi pesan ini "
            "aman dan tetap di jalur <b>Normal</b>.")
    st.markdown(
        f'<div class="d-result"><img src="{pic}" alt="">{stamp}<div class="d-lane">'
        '<div class="row"><span>START · Normal</span><span>Spam · FINISH</span></div>'
        f'<div class="d-track"><div class="d-flag"></div>'
        f'<div class="d-marker" style="left:calc({pct:.1f}% - {pct * 0.52:.1f}px)">{pct:.0f}%</div></div>'
        f'<p>{note}</p></div></div>',
        unsafe_allow_html=True)
