"""AquaRestore Studio — online-tool style underwater image restoration.

Structure (modelled on the iLovePDF experience)
-----------------------------------------------
* Top navigation ........ Home, logo, one link per tool
* Home .................. headline, filter pills, grid of tool cards
* Tool page ............. headline, subtitle, large red "Select image" button
                          with drag-and-drop
* Workspace ............. large canvas (left) + options panel with a large
                          red action button (right)
* Result pages .......... Quality analysis and Export

Tools: Fixed enhancement, Adaptive enhancement, Before / after, Compare,
Quality analysis, Export PNG.

The enhancement pipeline, quality-metric API and processing behaviour are
unchanged: ``enhance_image``, ``calculate_quality_metrics``,
``load_rgb_image``, ``get_metrics``, ``encode_png`` and ``process_image``
retain their original behaviour.
"""

import base64
import io
import json
import time
from html import escape

import numpy as np
import streamlit as st
from PIL import Image as PILImage
from PIL import ImageDraw
import streamlit.components.v1 as components

from aquarestore.contracts import calculate_quality_metrics
from aquarestore.enhancement.pipeline import enhance_image


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

def _favicon():
    """Draw a small brand mark so the tab icon matches the interface."""
    try:
        size = 64
        icon = PILImage.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(icon)
        draw.rounded_rectangle(
            (2, 2, size - 3, size - 3), radius=14, fill=(229, 50, 45, 255)
        )
        draw.ellipse((17, 17, 47, 47), outline=(255, 255, 255, 255), width=4)
        draw.line((23, 32, 41, 32), fill=(255, 255, 255, 255), width=4)
        return icon
    except Exception:
        return None


st.set_page_config(
    page_title="AquaRestore Studio",
    page_icon=_favicon(),
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

UPLOAD_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
MODES = ("Fixed", "Adaptive")
COMPARE_LAYOUTS = ("Original and enhanced", "Fixed and adaptive", "All three")
FILTERS = ("All", "Enhance", "Compare", "Inspect", "Export")

TOOLS = {
    "fixed": {
        "nav": "Fixed",
        "title": "Fixed enhancement",
        "subtitle": "Restore colour and contrast using preset pipeline parameters.",
        "desc": "Restore colour and contrast using preset pipeline parameters.",
        "cat": "Enhance",
        "color": "#f0704f",
        "glyph": "sliders_fixed",
    },
    "adaptive": {
        "nav": "Adaptive",
        "title": "Adaptive enhancement",
        "subtitle": "Restore colour and contrast with parameters adapted to each image.",
        "desc": "Enhance with pipeline parameters adapted to the content of each image.",
        "cat": "Enhance",
        "color": "#f0704f",
        "glyph": "sliders_adaptive",
    },
    "beforeafter": {
        "nav": "Before / after",
        "title": "Before / after",
        "subtitle": "Drag a divider to reveal the enhanced result over the original.",
        "desc": "Drag a divider to reveal the enhanced result over the original.",
        "cat": "Compare",
        "color": "#4d73c9",
        "glyph": "split",
    },
    "compare": {
        "nav": "Compare",
        "title": "Compare results",
        "subtitle": "View the original, fixed and adaptive results side by side.",
        "desc": "View the original, fixed and adaptive results side by side.",
        "cat": "Compare",
        "color": "#4d73c9",
        "glyph": "panels",
    },
    "analysis": {
        "nav": "Analysis",
        "title": "Quality analysis",
        "subtitle": "Measure luminance, contrast, saturation and edge strength.",
        "desc": "Measure luminance, contrast, saturation and edge strength.",
        "cat": "Inspect",
        "color": "#6aa84f",
        "glyph": "bars",
    },
    "export": {
        "nav": "Export",
        "title": "Export PNG",
        "subtitle": "Download lossless PNG files of the original and every result.",
        "desc": "Download lossless PNG files of the original and every result.",
        "cat": "Export",
        "color": "#2a9d8f",
        "glyph": "download",
    },
}

TITLE_HTML = {
    "fixed": "Fixed <em>enhancement</em>",
    "adaptive": "Adaptive <em>enhancement</em>",
    "beforeafter": "Before / <em>after</em>",
    "compare": "Compare <em>results</em>",
    "analysis": "Quality <em>analysis</em>",
    "export": "Export <em>PNG</em>",
}
for _tid, _html in TITLE_HTML.items():
    TOOLS[_tid]["title_html"] = _html

NAV = ["Home", "Fixed", "Adaptive", "Before / after", "Compare", "Analysis", "Export"]
NAV_TO_TOOL = {t["nav"]: tid for tid, t in TOOLS.items()}
NAV_TO_TOOL["Home"] = None

MODE_NOTES = {
    "Fixed": "Preset pipeline parameters, applied identically to every image.",
    "Adaptive": "Pipeline parameters adapted to the content of each image.",
}
MODE_TONE = {"Fixed": "fixed", "Adaptive": "adaptive"}

CORE_METRICS = (
    "Mean luminance",
    "Luminance contrast",
    "Mean saturation",
    "Edge strength",
)
EXTRA_METRICS = ("UIQM", "UCIQE")

PREVIEW_MAX_EDGE = 2400
PREVIEW_PNG_PIXELS = 2_000_000
PREVIEW_VERSION = 4
RESAMPLE = getattr(PILImage, "Resampling", PILImage).LANCZOS


# ---------------------------------------------------------
# DESIGN SYSTEM
# ---------------------------------------------------------

_EC = ':is(.element-container,[data-testid="stElementContainer"])'
_SEG = f"{_EC}:has(.seg-a)+{_EC}"
_LST = f"{_EC}:has(.lst-a)+{_EC}"
_PILL = f"{_EC}:has(.pill-a)+{_EC}"
_CTA = f"{_EC}:has(.cta-a)+{_EC}"
_TB = 'div[data-testid="stHorizontalBlock"]:has(.aq-brand)'
_WK = 'div[data-testid="stHorizontalBlock"]:has(.panel-head)'
_OV = (
    '[data-testid="stVerticalBlock"]:has(> '
    ':is(.element-container,[data-testid="stElementContainer"]) '
    ":is(.aq-brand,.tool-card))"
)

_ICON_STROKE = "%2347474f"


def _icon(body: str) -> str:
    return (
        'url("data:image/svg+xml;utf8,<svg xmlns=\'http://www.w3.org/2000/svg\' '
        "viewBox='0 0 16 16' fill='none' stroke='" + _ICON_STROKE + "' "
        "stroke-width='1.3' stroke-linecap='round' stroke-linejoin='round'>"
        + body
        + '</svg>") center/16px 16px no-repeat'
    )


ICON_LAYOUTS = [
    _icon("<rect x='1.5' y='3' width='5.5' height='10' rx='1.2'/><rect x='9' y='3' width='5.5' height='10' rx='1.2'/><path d='M11 8h1.6'/>"),
    _icon("<rect x='1.5' y='3' width='5.5' height='10' rx='1.2'/><rect x='9' y='3' width='5.5' height='10' rx='1.2'/>"),
    _icon("<rect x='1' y='3' width='4' height='10' rx='1'/><rect x='6' y='3' width='4' height='10' rx='1'/><rect x='11' y='3' width='4' height='10' rx='1'/>"),
]

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700;12..96,800&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&family=Fraunces:ital,wght@0,700;0,800;1,600;1,700&display=swap');

:root{
  --top:72px; --wh:calc(100vh - 120px);
  --bg:#f5f4f8; --paper:#fff; --soft:#f0eff5;
  --ink:#33333b; --ink2:#47474f; --muted:#5f5f69; --faint:#8d8d97;
  --line:#e4e3ec; --line2:#d3d2de;
  --red:#e5322d; --red2:#c9231f; --redtint:rgba(229,50,45,.08);
  --orig:#8a8a94; --fixed:#3f6fd1; --adaptive:#e5322d; --ok:#3f9a74;
  --sans:'DM Sans','Inter',-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  --display:'Bricolage Grotesque','DM Sans',-apple-system,'Segoe UI',Roboto,Arial,sans-serif;
  --serif:'Fraunces',Georgia,'Times New Roman',serif;
  color-scheme:light;
}

html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"],[data-testid="stMain"],section.main{ background:var(--bg)!important; color:var(--ink)!important; }
.stApp{
  background:
    radial-gradient(circle at 8% 14%, rgba(229,50,45,.045) 0, rgba(229,50,45,0) 34%),
    radial-gradient(circle at 96% 6%, rgba(229,50,45,.055) 0, rgba(229,50,45,0) 28%),
    var(--bg)!important;
}
html body .stApp{ font-family:var(--sans); -webkit-font-smoothing:antialiased; }
[data-testid="stMarkdownContainer"]{ color:var(--ink); }
[data-testid="stMarkdownContainer"] p{ margin:0; }

[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"],[data-testid="stHeader"],
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"],[data-testid="stExpandSidebarButton"],
#MainMenu,footer{ display:none!important; }
[data-stale="true"]{ opacity:1!important; transition:none!important; }

__EC__:has(style){ display:none!important; }
__EC__:has(.seg-a),__EC__:has(.lst-a),__EC__:has(.pill-a),__EC__:has(.cta-a){ display:none!important; }
[data-testid="stLayoutWrapper"]:has(.aq-brand){ position:absolute!important; width:0!important; height:0!important; margin:0!important; overflow:visible!important; }

.block-container{ max-width:none!important; padding:calc(var(--top) + 28px) 28px 24px 28px!important; }
[data-testid="stMain"] .block-container > [data-testid="stVerticalBlock"]{ gap:16px; }

/* ================= Top bar ================= */
__TB__{
  position:fixed!important; top:0; left:0; right:0; width:100vw!important; height:var(--top);
  z-index:1000000; align-items:center!important; gap:0!important; flex-wrap:nowrap!important;
  padding:0 28px!important; background:#fff; border-bottom:1px solid var(--line);
}
__TB__ > div{ min-width:0!important; }
__TB__ > div:nth-child(1){ flex:0 0 280px!important; width:280px!important; }
__TB__ > div:nth-child(2){ flex:1 1 auto!important; width:auto!important; }
__TB__ > div:nth-child(3){ flex:0 0 auto!important; width:auto!important; }
__TB__ [data-testid="stVerticalBlock"]{ gap:0!important; }
@media (max-width:980px){
  __TB__{ position:static!important; height:auto; flex-wrap:wrap!important; padding:10px 14px!important; width:100%!important; border-radius:12px; }
  __TB__ > div:nth-child(1),__TB__ > div:nth-child(2),__TB__ > div:nth-child(3){ flex:1 1 100%!important; width:100%!important; }
  [data-testid="stLayoutWrapper"]:has(.aq-brand){ position:static!important; width:auto!important; height:auto!important; }
  .block-container{ padding:16px 14px 24px 14px!important; }
}

.aq-brand{ display:flex; align-items:center; gap:11px; height:var(--top); }
.aq-brand img{ display:block; }
.aq-brand .bn{ font-family:var(--serif); font-weight:800; font-size:27px; letter-spacing:-.02em; line-height:1; color:#16161a; }
.aq-brand .bn b{ color:var(--red); font-weight:800; }
.tb-r{ display:flex; align-items:center; justify-content:flex-end; gap:9px; font-size:14.5px; font-weight:500; color:var(--muted); white-space:nowrap; height:var(--top); }
.dot{ width:8px; height:8px; border-radius:50%; background:var(--ok); display:inline-block; flex:none; }
.tb-r .dot{ box-shadow:0 0 0 4px rgba(63,154,116,.16); }
@media (max-width:1400px){
  __TB__ > div:nth-child(1){ flex-basis:240px!important; width:240px!important; }
  __TB__ label[data-baseweb="radio"]{ padding:0 12px!important; }
  __TB__ label[data-baseweb="radio"] p{ font-size:13.5px; }
}

__TB__ [role="radiogroup"]{ display:flex!important; flex-direction:row!important; flex-wrap:nowrap!important; gap:0!important; align-items:stretch; }
__TB__ label[data-baseweb="radio"]{ height:var(--top); padding:0 20px!important; position:relative; border-radius:0; transition:background .15s ease; }
__TB__ label[data-baseweb="radio"] p{ font-family:var(--display); font-size:15px; font-weight:600; letter-spacing:.02em; text-transform:uppercase; color:var(--ink); transition:color .15s ease; }
__TB__ label[data-baseweb="radio"]:hover p{ color:var(--red); }
__TB__ > div:nth-child(3) [data-testid="stButton"]{ width:auto; }
__TB__ > div:nth-child(3) button{ width:auto!important; min-height:42px!important; padding:0 22px!important; border-radius:10px!important; font-size:15px!important; }

/* overlay buttons (tool cards, logo) */
__OV__{ position:relative; }
__OV__ > :is(.element-container,[data-testid="stElementContainer"]):has([data-testid="stButton"]){ position:absolute!important; inset:0!important; width:100%!important; height:100%!important; margin:0!important; z-index:3; }
__OV__ [data-testid="stButton"]{ height:100%!important; width:100%!important; }
__OV__ [data-testid="stButton"] button{ height:100%!important; width:100%!important; min-height:0!important; opacity:0; cursor:pointer; border-radius:24px!important; }

/* radios: shared reset */
[data-testid="stRadio"] [data-testid="stWidgetLabel"]{ display:none; }
[data-testid="stRadio"] label[data-baseweb="radio"] > *:first-child:not(input):not(:has(p)){ display:none!important; }
[data-testid="stRadio"] label[data-baseweb="radio"]{ margin:0!important; cursor:pointer; align-items:center; }
[data-testid="stRadio"] label[data-baseweb="radio"] p{ margin:0; display:inline-flex; align-items:center; white-space:nowrap; line-height:1.2; }
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:focus-visible){ outline:2px solid var(--red); outline-offset:2px; }

/* radios: filter pills */
__PILL__{ display:flex; justify-content:center; }
__PILL__ [data-testid="stRadio"],__PILL__ [data-testid="stRadio"] > div{ width:auto!important; }
__PILL__ [role="radiogroup"]{ display:flex!important; flex-wrap:wrap!important; justify-content:center; gap:14px!important; }
__PILL__ label[data-baseweb="radio"]{ padding:11px 24px!important; border:1px solid var(--line2); border-radius:999px; background:rgba(255,255,255,.7); transition:background .15s ease, border-color .15s ease, box-shadow .15s ease; }
__PILL__ label[data-baseweb="radio"] p{ font-size:17px; font-weight:500; color:var(--ink); }
__PILL__ label[data-baseweb="radio"]:hover p{ color:var(--red); }
__PILL__ label[data-baseweb="radio"]:hover{ background:#fff; border-color:#b9b8c6; box-shadow:0 2px 6px rgba(40,40,60,.06); }

/* radios: segmented control (options panel) */
__SEG__ [data-testid="stRadio"],__SEG__ [data-testid="stRadio"] > div{ width:100%!important; }
__SEG__ [role="radiogroup"]{ display:flex!important; flex-direction:row!important; flex-wrap:nowrap!important; width:100%!important; gap:3px!important; padding:4px; background:var(--soft); border-radius:12px; }
__SEG__ label[data-baseweb="radio"]{ flex:1 1 0!important; min-width:0; justify-content:center; padding:10px 8px!important; border-radius:9px; transition:background .15s ease, box-shadow .15s ease; }
__SEG__ label[data-baseweb="radio"] p{ font-size:14.5px; font-weight:500; color:var(--ink2); }
__SEG__ label[data-baseweb="radio"]:hover{ background:rgba(255,255,255,.6); }
__SEG__ label[data-baseweb="radio"] p::before{ content:""; width:8px; height:8px; border-radius:50%; margin-right:8px; flex:none; }
__SEG__ label[data-baseweb="radio"]:nth-child(1) p::before{ background:var(--fixed); }
__SEG__ label[data-baseweb="radio"]:nth-child(2) p::before{ background:var(--adaptive); }

/* radios: option list (options panel) */
__LST__ [role="radiogroup"]{ display:flex!important; flex-direction:column!important; gap:4px!important; width:100%; }
__LST__ label[data-baseweb="radio"]{ padding:11px 12px!important; border-radius:10px; border:1px solid transparent; transition:background .15s ease, border-color .15s ease; }
__LST__ label[data-baseweb="radio"] p{ font-size:14.5px; font-weight:500; color:var(--ink2); }
__LST__ label[data-baseweb="radio"]:hover{ background:var(--soft); }
__LST__ label[data-baseweb="radio"] p::before{ content:""; width:16px; height:16px; margin-right:11px; flex:none; }
__LST__ label[data-baseweb="radio"]:nth-child(1) p::before{ background:ICON_L1; }
__LST__ label[data-baseweb="radio"]:nth-child(2) p::before{ background:ICON_L2; }
__LST__ label[data-baseweb="radio"]:nth-child(3) p::before{ background:ICON_L3; }

/* file uploader: large select button */
[data-testid="stFileUploader"]{ max-width:780px; margin:0 auto; }
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"]{ display:none; }
[data-testid="stFileUploaderDropzone"]{
  display:flex!important; flex-direction:column!important; align-items:center!important; justify-content:center!important; gap:20px!important;
  min-height:340px; padding:40px 20px!important; background:rgba(255,255,255,.6)!important; border:2px dashed var(--line2)!important; border-radius:26px!important;
  transition:border-color .18s ease, background-color .18s ease, box-shadow .18s ease;
}
[data-testid="stFileUploaderDropzone"]:hover,[data-testid="stFileUploaderDropzone"]:focus-within{ border-color:var(--red)!important; background:#fff!important; box-shadow:0 0 0 6px var(--redtint); }
[data-testid="stFileUploaderDropzoneInstructions"]{ display:none!important; }
[data-testid="stFileUploaderDropzone"] button{
  width:min(420px,100%)!important; height:100px!important; padding:0!important; border:0!important; border-radius:14px!important; background:var(--red)!important;
  box-shadow:0 10px 22px -8px rgba(229,50,45,.6)!important; display:flex; align-items:center; justify-content:center; transition:background .15s ease, transform .1s ease;
}
[data-testid="stFileUploaderDropzone"] button *{ display:none!important; }
[data-testid="stFileUploaderDropzone"] button::after{ content:"Select image"; font-family:var(--display); font-size:31px; font-weight:700; letter-spacing:-.015em; line-height:1; color:#fff; }
[data-testid="stFileUploaderDropzone"] button:hover{ background:var(--red2)!important; }
[data-testid="stFileUploaderDropzone"] button:active{ transform:translateY(1px); }
[data-testid="stFileUploaderDropzone"] button:focus-visible{ outline:3px solid var(--ink); outline-offset:3px; }
[data-testid="stFileUploaderDropzone"]::after{ content:"or drop an image here"; font-family:var(--serif); font-style:italic; font-weight:600; font-size:20px; color:var(--red); }
[data-testid="stFileUploaderFile"],[data-testid="stFileUploaderPagination"]{ display:none!important; }

/* buttons */
.stButton,.stDownloadButton,[data-testid="stButton"],[data-testid="stDownloadButton"]{ width:100%; }
.stButton button,.stDownloadButton button,[data-testid="stButton"] button,[data-testid="stDownloadButton"] button{
  width:100%; min-height:48px; border-radius:12px; border:1px solid var(--line2); background:#fff; color:var(--ink);
  font-size:16px; font-weight:600; transition:background .15s ease, border-color .15s ease, transform .1s ease, box-shadow .15s ease;
}
.stButton button:hover,.stDownloadButton button:hover,[data-testid="stButton"] button:hover,[data-testid="stDownloadButton"] button:hover{ border-color:var(--ink2); color:var(--ink); background:#fff; }
.stButton button:active,.stDownloadButton button:active{ transform:translateY(1px); }
.stButton button:focus-visible,.stDownloadButton button:focus-visible,[data-testid="stButton"] button:focus-visible,[data-testid="stDownloadButton"] button:focus-visible{ outline:3px solid var(--ink); outline-offset:2px; }
.stButton button p,.stDownloadButton button p,[data-testid="stButton"] button p,[data-testid="stDownloadButton"] button p{ color:inherit; font-size:inherit; font-weight:inherit; }
button[kind="primary"],button[data-testid="stBaseButton-primary"]{ background:var(--red)!important; border-color:var(--red)!important; color:#fff!important; box-shadow:0 8px 18px -8px rgba(229,50,45,.65)!important; }
button[kind="primary"]:hover,button[data-testid="stBaseButton-primary"]:hover{ background:var(--red2)!important; border-color:var(--red2)!important; color:#fff!important; }
button[kind="primary"] p,button[data-testid="stBaseButton-primary"] p{ color:#fff!important; }
[data-testid="stSpinner"] *{ color:var(--muted)!important; font-size:14px; }

/* ================= Hero ================= */
.hero{ text-align:center; padding:18px 0 4px; }
.hero .eb{ display:inline-flex; align-items:center; gap:12px; margin-bottom:16px; font-size:13px; font-weight:700; letter-spacing:.18em; text-transform:uppercase; color:var(--red); }
.hero .eb::before,.hero .eb::after{ content:""; width:34px; height:2px; border-radius:2px; background:var(--red); opacity:.55; }
.hero .ht{ font-family:var(--display); font-size:clamp(32px,4.4vw,60px); font-weight:800; letter-spacing:-.03em; line-height:1.08; color:var(--ink); }
.hero .ht em,.panel-head .pk em{ font-family:var(--serif); font-style:italic; font-weight:700; letter-spacing:-.02em; color:var(--red); }
.hero .hs{ font-size:clamp(16px,1.65vw,23px); line-height:1.55; color:var(--ink2); max-width:920px; margin:16px auto 0; }
.hero .hc{ display:inline-flex; align-items:center; gap:9px; margin-top:18px; padding:7px 16px; border:1px solid var(--line2); border-radius:999px; background:rgba(255,255,255,.75); font-size:14px; font-weight:500; color:var(--ink2); }
.hero.tight .ht{ font-size:clamp(30px,3.8vw,50px); }
.upnote{ text-align:center; font-size:14px; color:var(--muted); margin-top:4px; }

/* ================= Tool cards ================= */
.tool-card{ position:relative; min-height:288px; padding:30px 30px 26px; background:#fff; border:1px solid var(--line); border-radius:24px; transition:box-shadow .2s ease, transform .2s ease, border-color .2s ease; }
.tool-card .ti{ display:block; width:60px; height:60px; margin-bottom:26px; }
.tool-card .tc{ position:absolute; top:34px; right:30px; font-size:11.5px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:var(--red); }
.tool-card .tt{ font-family:var(--display); font-size:25px; font-weight:700; letter-spacing:-.02em; color:var(--ink); line-height:1.2; transition:color .15s ease; }
.tool-card .td{ font-size:16px; line-height:1.55; color:var(--muted); margin-top:12px; }
__OV__:hover .tool-card .tt{ color:var(--red); }
__OV__:hover .tool-card{ box-shadow:0 14px 34px -14px rgba(40,40,70,.28); border-color:var(--line2); transform:translateY(-2px); }
__OV__:has(button:focus-visible) .tool-card{ outline:3px solid var(--ink); outline-offset:2px; }

.aq-foot span:first-child{ color:var(--ink2); font-weight:500; }
.aq-foot{ margin-top:40px; padding:18px 4px 6px; border-top:1px solid var(--line); font-size:14px; color:var(--muted); display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; }

/* ================= Workspace ================= */
__WK__{ gap:20px!important; flex-wrap:nowrap!important; align-items:flex-start!important; }
__WK__ > div:nth-child(1){ flex:1 1 0!important; width:auto!important; min-width:0!important; }
__WK__ > div:nth-child(1) __EC__{ height:var(--wh)!important; }
__WK__ > div:nth-child(1) iframe{ height:var(--wh)!important; width:100%; border:0; display:block; }
__WK__ > div:nth-child(2){
  flex:0 0 390px!important; width:390px!important; min-width:390px!important; height:var(--wh); overflow-y:auto; padding:0 24px!important;
  background:#fff; border:1px solid var(--line); border-radius:20px; scrollbar-width:thin; scrollbar-color:var(--line2) transparent; display:flex; flex-direction:column;
}
__WK__ > div:nth-child(2) > [data-testid="stVerticalBlock"]{ flex:1; gap:0!important; display:flex; flex-direction:column; }
__CTA__{ margin-top:auto; position:sticky; bottom:0; z-index:2; background:linear-gradient(180deg,rgba(255,255,255,0) 0,#fff 22%); padding:22px 0 20px; }
__CTA__ button{ min-height:60px!important; font-size:19px!important; border-radius:14px!important; }
__CTA__ button p{ font-family:var(--display); font-weight:700; letter-spacing:-.005em; }
@media (max-width:1100px){
  __WK__{ flex-wrap:wrap!important; }
  __WK__ > div:nth-child(1){ flex:1 1 100%!important; }
  __WK__ > div:nth-child(2){ flex:1 1 100%!important; width:100%!important; min-width:0!important; height:auto; }
  __WK__ > div:nth-child(1) __EC__,__WK__ > div:nth-child(1) iframe{ height:68vh!important; }
}

.panel-head{ padding:24px 0 18px; border-bottom:1px solid var(--line); }
.panel-head .pe{ font-size:12px; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:var(--red); margin-bottom:8px; }
.panel-head .pk{ font-family:var(--display); font-size:27px; font-weight:800; letter-spacing:-.025em; color:var(--ink); line-height:1.15; }
.panel-head .pdsc{ font-size:14.5px; line-height:1.55; color:var(--muted); margin-top:8px; }
.plabel{ font-size:12px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:var(--red); padding:20px 0 10px; }
.pnote{ font-size:13px; line-height:1.55; color:var(--muted); padding-top:10px; }

.psec{ padding:20px 0 22px; border-top:1px solid var(--line); margin-top:20px; }
.psec.first{ border-top:0; margin-top:0; }
.ih{ display:flex; align-items:center; justify-content:space-between; margin-bottom:14px; font-size:12px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }
.ih .hm{ display:inline-flex; align-items:center; gap:7px; letter-spacing:0; text-transform:none; font-size:13.5px; font-weight:600; color:var(--ink); }
.tdot{ width:9px; height:9px; border-radius:50%; display:inline-block; flex:none; }
.t-orig{ background:var(--orig); } .t-fixed{ background:var(--fixed); } .t-adaptive{ background:var(--adaptive); } .t-ink{ background:var(--ink); }
.fname{ font-family:var(--display); font-size:19px; font-weight:700; letter-spacing:-.01em; color:var(--ink); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; margin-bottom:10px; }
.kv{ display:flex; justify-content:space-between; align-items:baseline; gap:14px; padding:5px 0; font-size:14.5px; }
.kv .k{ color:var(--muted); }
.kv .v{ color:var(--ink); font-weight:600; text-align:right; font-variant-numeric:tabular-nums; }
.hist{ display:block; width:100%; height:auto; border-radius:10px; }
.hax{ display:flex; justify-content:space-between; margin-top:8px; font-size:12px; color:var(--muted); }
.mlist > .mb{ padding:14px 0; border-top:1px solid var(--line); }
.mlist > .mb:first-child{ border-top:0; padding-top:0; }
.mn{ font-size:14.5px; font-weight:600; color:var(--ink); margin-bottom:10px; }
.mv{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }
.mv .c{ display:flex; flex-direction:column; gap:4px; min-width:0; }
.mv .cl{ font-size:11.5px; font-weight:600; color:var(--muted); }
.mv .cv{ font-size:15px; font-weight:700; color:var(--ink); font-variant-numeric:tabular-nums; }
.mv .cb{ height:4px; border-radius:2px; background:var(--soft); overflow:hidden; margin-top:2px; }
.mv .cb i{ display:block; height:100%; border-radius:2px; }
.note{ font-size:13px; line-height:1.55; color:var(--muted); }

/* ================= Processing and error ================= */
.proc{ text-align:center; padding:12vh 0 8vh; }
.proc .pt{ font-family:var(--display); font-size:clamp(30px,3.8vw,50px); font-weight:800; letter-spacing:-.03em; color:var(--ink); }
.proc .ps{ font-size:18px; color:var(--ink2); margin-top:14px; }
.lbar{ position:relative; width:min(420px,80%); height:8px; border-radius:6px; background:#e4e3ec; overflow:hidden; margin:34px auto 0; }
.lbar i{ position:absolute; top:0; bottom:0; left:0; width:38%; background:var(--red); border-radius:6px; animation:lb 1.2s ease-in-out infinite; }
@keyframes lb{ 0%{ transform:translateX(-100%); } 100%{ transform:translateX(270%); } }
.err{ max-width:680px; margin:8vh auto 0; padding:30px 32px; background:#fff; border:1px solid var(--line); border-left:5px solid var(--red); border-radius:18px; }
.err .et{ font-family:var(--display); font-size:27px; font-weight:800; letter-spacing:-.02em; color:var(--ink); }
.err .em{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:13px; line-height:1.55; color:var(--ink2); background:var(--bg); border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin:16px 0; word-break:break-word; }
.err .eh{ font-size:15px; line-height:1.6; color:var(--muted); }

/* ================= Analysis ================= */
.kpis{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-top:24px; }
@media (max-width:900px){ .kpis{ grid-template-columns:repeat(2,1fr); } }
.kpi{ --t:var(--ink); padding:20px 22px; background:#fff; border:1px solid var(--line); border-top:4px solid var(--t); border-radius:18px; }
.kpi.k-orig{ --t:var(--orig); } .kpi.k-fixed{ --t:var(--fixed); } .kpi.k-adaptive{ --t:var(--adaptive); }
.kpi .kl{ font-size:12px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:var(--red); }
.kpi .kn{ font-family:var(--display); font-size:34px; font-weight:800; letter-spacing:-.03em; color:var(--ink); margin-top:8px; line-height:1.1; }
.kpi .ks{ font-size:13.5px; color:var(--muted); margin-top:4px; }

.card{ background:#fff; border:1px solid var(--line); border-radius:20px; }
.gt{ overflow-x:auto; margin-top:16px; }
.gr{ display:grid; grid-template-columns:minmax(170px,1.1fr) repeat(3,minmax(140px,1fr)); min-width:600px; }
.gr > div{ padding:18px 24px; border-bottom:1px solid var(--line); }
.gr:last-child > div{ border-bottom:0; }
.gr.gh > div{ display:flex; align-items:center; gap:9px; padding:16px 24px; font-size:13px; font-weight:700; color:var(--muted); background:#faf9fc; }
.gr:not(.gh):hover > div{ background:#fcfbfe; }
.gr .mname{ font-family:var(--display); font-size:17px; font-weight:700; letter-spacing:-.01em; color:var(--ink); }
.av{ font-family:var(--display); font-size:21px; font-weight:700; color:var(--ink); font-variant-numeric:tabular-nums; }
.ad{ font-size:13px; color:var(--muted); margin-top:3px; }
.ab{ height:6px; background:var(--soft); border-radius:3px; margin-top:12px; overflow:hidden; }
.ab i{ display:block; height:100%; border-radius:3px; }
.ab i.o{ background:var(--orig); } .ab i.f{ background:var(--fixed); } .ab i.a{ background:var(--adaptive); }
.ag{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:16px; }
@media (max-width:900px){ .ag{ grid-template-columns:1fr; } }
.pc{ padding:24px 26px; }
.pc .ih{ margin-bottom:16px; }
.rt{ display:flex; align-items:center; justify-content:space-between; font-size:16px; margin-bottom:8px; }
.rt span:first-child{ display:inline-flex; align-items:center; gap:9px; }
.rt .rv{ font-weight:700; font-variant-numeric:tabular-nums; }
.pc .note{ margin-top:14px; }

/* ================= Export ================= */
.xc{ padding:16px 16px 20px; overflow:hidden; }
.xth{ position:relative; width:100%; aspect-ratio:16/10; border-radius:12px; overflow:hidden; background:var(--soft); margin-bottom:16px; }
.xth img{ width:100%; height:100%; object-fit:cover; display:block; }
.xth::after{ content:""; position:absolute; left:0; right:0; bottom:0; height:4px; background:var(--t,var(--ink)); }
.xt{ display:flex; align-items:center; justify-content:space-between; gap:8px; padding:0 6px; font-family:var(--display); font-size:21px; font-weight:700; letter-spacing:-.015em; color:var(--ink); }
.xt span:first-child{ display:inline-flex; align-items:center; gap:10px; }
.xd{ font-size:14.5px; line-height:1.5; color:var(--muted); margin:6px 6px 12px; }
.xc .kv{ padding:5px 6px; }
.xb{ font-size:11px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--red); background:var(--redtint); border-radius:6px; padding:4px 8px; }

@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{ transition-duration:.01ms!important; animation-duration:.01ms!important; }
}
"""

CSS = (
    CSS.replace("__SEG__", _SEG)
    .replace("__LST__", _LST)
    .replace("__PILL__", _PILL)
    .replace("__CTA__", _CTA)
    .replace("__TB__", _TB)
    .replace("__WK__", _WK)
    .replace("__OV__", _OV)
    .replace("__EC__", _EC)
    .replace("ICON_L1", ICON_LAYOUTS[0])
    .replace("ICON_L2", ICON_LAYOUTS[1])
    .replace("ICON_L3", ICON_LAYOUTS[2])
)

CENTERED_CSS = (
    ".block-container{max-width:1560px!important;margin-left:auto!important;"
    "margin-right:auto!important;}"
)


def nav_css(nav: str) -> str:
    index = NAV.index(nav) + 1
    return (
        f'<style>{_TB} [role="radiogroup"] > label:nth-child({index}) p'
        "{color:var(--red)!important}</style>"
    )


def pill_css(filter_name: str) -> str:
    index = FILTERS.index(filter_name) + 1
    return (
        f'<style>{_PILL} [role="radiogroup"] > label:nth-child({index})'
        "{background:#2e2e35!important;border-color:#2e2e35!important}"
        f'{_PILL} [role="radiogroup"] > label:nth-child({index}) p'
        "{color:#fff!important}</style>"
    )


def seg_css(mode: str) -> str:
    index = MODES.index(mode) + 1
    return (
        f'<style>{_SEG} [role="radiogroup"] > label:nth-child({index})'
        "{background:#fff!important;box-shadow:0 1px 3px rgba(40,40,70,.18),"
        "0 0 0 1px var(--line2)!important}"
        f'{_SEG} [role="radiogroup"] > label:nth-child({index}) p'
        "{color:var(--ink)!important;font-weight:700!important}</style>"
    )


def list_css(layout: str) -> str:
    index = COMPARE_LAYOUTS.index(layout) + 1
    return (
        f'<style>{_LST} [role="radiogroup"] > label:nth-child({index})'
        "{background:var(--redtint)!important;border-color:rgba(229,50,45,.35)!important}"
        f'{_LST} [role="radiogroup"] > label:nth-child({index}) p'
        "{color:var(--ink)!important;font-weight:700!important}</style>"
    )


def _compact(markup: str) -> str:
    """Collapse markup to one line so Markdown never sees code blocks."""
    return " ".join(
        line.strip() for line in markup.strip().splitlines() if line.strip()
    )


def md(markup: str) -> None:
    """Render trusted HTML through Streamlit."""
    st.markdown(_compact(markup), unsafe_allow_html=True)


def svg_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(
        svg.encode("utf-8")
    ).decode("ascii")


LOGO = svg_uri(
    '<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" '
    'viewBox="0 0 36 36"><rect width="36" height="36" rx="9" fill="#e5322d"/>'
    '<circle cx="18" cy="18" r="10" fill="none" stroke="#fff" stroke-width="2.4"/>'
    '<path d="M10.5 18c2.5-2.6 5-2.6 7.5 0s5 2.6 7.5 0" fill="none" '
    'stroke="#fff" stroke-width="2.4" stroke-linecap="round"/></svg>'
)

GLYPHS = {
    "sliders_fixed": (
        "<path d='M30 33h22M30 41h22M30 49h22'/>"
        "<circle cx='37' cy='33' r='3' fill='#fff' stroke='none'/>"
        "<circle cx='37' cy='41' r='3' fill='#fff' stroke='none'/>"
        "<circle cx='37' cy='49' r='3' fill='#fff' stroke='none'/>"
    ),
    "sliders_adaptive": (
        "<path d='M30 33h22M30 41h22M30 49h22'/>"
        "<circle cx='36' cy='33' r='3' fill='#fff' stroke='none'/>"
        "<circle cx='46' cy='41' r='3' fill='#fff' stroke='none'/>"
        "<circle cx='40' cy='49' r='3' fill='#fff' stroke='none'/>"
    ),
    "split": (
        "<rect x='29' y='30' width='24' height='22' rx='3'/><path d='M41 30v22'/>"
        "<path d='M41 30h9a3 3 0 0 1 3 3v16a3 3 0 0 1-3 3h-9z' fill='#fff' stroke='none'/>"
    ),
    "panels": (
        "<rect x='28' y='31' width='11' height='20' rx='2.5'/>"
        "<rect x='43' y='31' width='11' height='20' rx='2.5'/>"
    ),
    "bars": "<path d='M32 52V43M41 52V34M50 52V46' stroke-width='4.5'/>",
    "download": "<path d='M41 30v17M34 40l7 7 7-7M31 54h20'/>",
}


def tool_icon(tool_id: str) -> str:
    tool = TOOLS[tool_id]
    colour = tool["color"]
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='60' height='60' "
        "viewBox='0 0 60 60'>"
        f"<rect x='0' y='0' width='38' height='38' rx='8' fill='{colour}' "
        "fill-opacity='.3'/>"
        f"<rect x='22' y='22' width='38' height='38' rx='8' fill='{colour}'/>"
        "<g fill='none' stroke='#fff' stroke-width='2.6' stroke-linecap='round' "
        f"stroke-linejoin='round'>{GLYPHS[tool['glyph']]}</g></svg>"
    )
    return svg_uri(svg)


# ---------------------------------------------------------
# VIEWER COMPONENT (canvas)
# ---------------------------------------------------------

VIEWER_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700;800&family=DM+Sans:wght@400;500;600;700&display=swap');
:root{--ink:#33333b;--ink2:#47474f;--muted:#5f5f69;--line:#e4e3ec;--line2:#d3d2de;--red:#e5322d;--soft:#f0eff5}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden;background:transparent;color:var(--ink);font-family:'DM Sans','Inter',-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
.shell{height:100%;display:flex;flex-direction:column;background:#fff;border:1px solid var(--line);border-radius:20px;overflow:hidden}
.bar{flex:none;height:56px;display:flex;align-items:center;gap:12px;padding:0 12px 0 22px;border-bottom:1px solid var(--line);background:#fff}
.dt{flex:none;width:10px;height:10px;border-radius:50%;background:var(--ink)}
.ttl{flex:0 1 auto;min-width:0;font-family:'Bricolage Grotesque','DM Sans',sans-serif;font-size:17px;font-weight:700;letter-spacing:-.01em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sub{flex:none;font-size:14px;color:var(--muted);white-space:nowrap}
.sub::before{content:"";display:inline-block;width:1px;height:14px;background:var(--line2);margin-right:12px;vertical-align:-2px}
.sp{flex:1}
.grp{display:flex;align-items:center;gap:2px}
.sep{width:1px;height:22px;background:var(--line);margin:0 6px;flex:none}
.btn{position:relative;height:38px;min-width:38px;padding:0 10px;display:inline-flex;align-items:center;justify-content:center;gap:7px;border:1px solid transparent;border-radius:10px;background:transparent;color:var(--ink2);font:inherit;font-size:14px;font-weight:600;cursor:pointer;transition:background .15s,color .15s,border-color .15s}
.btn:hover{background:var(--soft);color:var(--ink)}
.btn:active{background:#e6e5ee}
.btn:focus-visible{outline:2px solid var(--ink);outline-offset:1px}
.btn[aria-pressed="true"]{background:rgba(229,50,45,.09);color:var(--red);border-color:rgba(229,50,45,.32)}
.btn svg{width:19px;height:19px;fill:none;stroke:currentColor;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round}
.btn[data-tip]::after{content:attr(data-tip);position:absolute;top:calc(100% + 9px);left:50%;transform:translate(-50%,-2px);background:#26262d;color:#fff;font-size:12.5px;font-weight:500;line-height:1;padding:8px 11px;border-radius:8px;white-space:nowrap;pointer-events:none;opacity:0;transition:opacity .14s,transform .14s;z-index:30;box-shadow:0 8px 20px rgba(0,0,0,.25)}
.btn[data-tip]:hover::after,.btn[data-tip]:focus-visible::after{opacity:1;transform:translate(-50%,0);transition-delay:.4s}
.btn.edge[data-tip]::after{left:auto;right:0;transform:translate(0,-2px)}
.btn.edge[data-tip]:hover::after,.btn.edge[data-tip]:focus-visible::after{transform:translate(0,0)}
.zl{min-width:54px;text-align:center;font-size:14px;font-weight:700;font-variant-numeric:tabular-nums}
.stage{position:relative;flex:1;min-height:0;display:flex;background:#ecebf2;cursor:grab;outline:none;touch-action:none}
.stage.dragging{cursor:grabbing}
.stage:focus-visible{box-shadow:inset 0 0 0 3px rgba(229,50,45,.5)}
.panel{position:relative;flex:1 1 0;min-width:0;overflow:hidden}
.panel+.panel{border-left:2px solid #fff}
.layer{position:absolute;inset:0;overflow:hidden}
.layer img{position:absolute;left:0;top:0;max-width:none;transform-origin:0 0;user-select:none;-webkit-user-drag:none;opacity:0;transition:opacity .3s ease;will-change:transform}
.layer img.ready{opacity:1}
.tag{position:absolute;top:14px;z-index:3;max-width:calc(50% - 32px);display:inline-flex;align-items:center;padding:8px 13px;font-size:13px;font-weight:700;line-height:1;color:var(--ink);background:rgba(255,255,255,.94);border-radius:999px;box-shadow:0 2px 8px rgba(30,30,60,.18);pointer-events:none;overflow:hidden;white-space:nowrap}
.tag i{flex:none;width:9px;height:9px;border-radius:50%;margin-right:8px}
.tag .tx{overflow:hidden;text-overflow:ellipsis}
.tag.l{left:14px}.tag.r{right:14px}
.divider{position:absolute;top:0;bottom:0;left:50%;width:0;z-index:5}
.divider .line{position:absolute;top:0;bottom:0;left:-1.5px;width:3px;background:#fff;box-shadow:0 0 0 1px rgba(0,0,0,.25)}
.divider .hit{position:absolute;top:0;bottom:0;left:-16px;width:32px;cursor:ew-resize;touch-action:none}
.grip{position:absolute;top:50%;left:0;width:44px;height:44px;margin:-22px 0 0 -22px;border-radius:50%;background:var(--red);color:#fff;border:3px solid #fff;box-shadow:0 4px 14px rgba(0,0,0,.35);display:grid;place-items:center;cursor:ew-resize;transition:box-shadow .15s,transform .15s}
.grip:hover{box-shadow:0 6px 18px rgba(0,0,0,.45)}
.grip:focus-visible{outline:3px solid var(--ink);outline-offset:3px}
.divider.on .grip{transform:scale(1.08)}
.grip svg{width:19px;height:19px;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.msg{position:absolute;inset:0;display:none;place-items:center;text-align:center;padding:24px;font-size:15px;color:var(--muted);z-index:6}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head>
<body>
<div class="shell">
  <div class="bar">
    <i class="dt" id="dt"></i><span class="ttl" id="ttl"></span><span class="sub" id="sub"></span><span class="sp"></span>
    <div class="grp" id="gOrig">
      <button class="btn" id="bOrig" aria-pressed="false" data-tip="Show original &nbsp;\">
        <svg viewBox="0 0 24 24"><path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="2.8"/></svg><span>Original</span>
      </button>
    </div>
    <span class="sep" id="sOrig"></span>
    <div class="grp">
      <button class="btn" id="bOut" aria-label="Zoom out" data-tip="Zoom out &nbsp;&minus;"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.2-4.2M8 11h6"/></svg></button>
      <span class="zl" id="zl">100%</span>
      <button class="btn" id="bIn" aria-label="Zoom in" data-tip="Zoom in &nbsp;+"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.2-4.2M11 8v6M8 11h6"/></svg></button>
    </div>
    <span class="sep"></span>
    <div class="grp">
      <button class="btn" id="bFit" aria-label="Fit to screen" data-tip="Fit to screen &nbsp;0"><svg viewBox="0 0 24 24"><path d="M4 9V5a1 1 0 0 1 1-1h4M15 4h4a1 1 0 0 1 1 1v4M20 15v4a1 1 0 0 1-1 1h-4M9 20H5a1 1 0 0 1-1-1v-4"/></svg></button>
      <button class="btn" id="b100" aria-label="Actual size" data-tip="Actual size &nbsp;1">1:1</button>
    </div>
    <span class="sep"></span>
    <div class="grp">
      <button class="btn edge" id="bReset" aria-label="Reset view" data-tip="Reset view"><svg viewBox="0 0 24 24"><path d="M4 12a8 8 0 1 0 2.6-5.9"/><path d="M4 4v4.5h4.5"/></svg></button>
    </div>
  </div>
  <div class="stage" id="stage" tabindex="0" aria-label="Image canvas"></div>
</div>
<script>
(function(){
const cfg=__CONFIG__;
const $=s=>document.querySelector(s);
const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
const TONES={orig:'#8a8a94',fixed:'#3f6fd1',adaptive:'#e5322d',ink:'#33333b'};
const stage=$('#stage'),zl=$('#zl'),bOrig=$('#bOrig');
$('#ttl').textContent=cfg.title;$('#sub').textContent=cfg.sub;$('#dt').style.background=TONES[cfg.tone]||TONES.ink;
const pw=cfg.pw,ph=cfg.ph,k=cfg.pw/cfg.ow;
let s=1,cx=pw/2,cy=ph/2,fitMode=true,d=0.5,showOrig=false;
const views=[];let la=null,divider=null,grip=null,tagEl=null,altImg=null;
const ARROWS='<svg viewBox="0 0 24 24"><path d="M9 7l-5 5 5 5M15 7l5 5-5 5"/></svg>';
function el(t,c){const e=document.createElement(t);if(c)e.className=c;return e}
function setTag(t,txt,tone){t.querySelector('i').style.background=TONES[tone]||TONES.ink;t.querySelector('.tx').textContent=txt}
function tag(txt,side,tone){const t=el('span','tag '+side);const i=el('i');const x=el('span','tx');t.append(i,x);setTag(t,txt,tone);return t}
const msg=el('div','msg');msg.textContent='The preview could not be displayed.';
function mkImg(src){const i=new Image();i.draggable=false;i.alt='';i.style.width=pw+'px';i.style.height=ph+'px';i.onload=()=>i.classList.add('ready');i.onerror=()=>{msg.style.display='grid'};i.src=src;return i}

if(cfg.layout==='split'){
  cfg.panels.forEach(p=>{const pn=el('div','panel'),ly=el('div','layer'),im=mkImg(p.src);ly.appendChild(im);pn.append(ly,tag(p.label,'l',p.tone));stage.appendChild(pn);views.push({el:pn,imgs:[im]})});
}else if(cfg.layout==='slider'){
  const pn=el('div','panel'),lb=el('div','layer'),ia=mkImg(cfg.panels[0].src),ib=mkImg(cfg.panels[1].src);
  lb.appendChild(ia);la=el('div','layer');la.appendChild(ib);
  divider=el('div','divider');
  divider.innerHTML='<div class="line"></div><div class="hit"></div><button class="grip" id="grip" role="slider" aria-label="Comparison divider" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50">'+ARROWS+'</button>';
  pn.append(lb,la,tag(cfg.panels[0].label,'l',cfg.panels[0].tone),tag(cfg.panels[1].label,'r',cfg.panels[1].tone),divider);
  stage.appendChild(pn);views.push({el:pn,imgs:[ia,ib]});grip=divider.querySelector('#grip');
}else{
  const pn=el('div','panel'),ly=el('div','layer'),bi=mkImg(cfg.panels[0].src);ly.appendChild(bi);
  const imgs=[bi];
  if(cfg.alt){altImg=mkImg(cfg.alt.src);altImg.style.visibility='hidden';ly.appendChild(altImg);imgs.push(altImg)}
  tagEl=tag(cfg.panels[0].label,'l',cfg.panels[0].tone);pn.append(ly,tagEl);stage.appendChild(pn);views.push({el:pn,imgs:imgs});
}
stage.appendChild(msg);
if(!altImg){$('#gOrig').remove();$('#sOrig').remove()}

function fmtPct(p){return (p<10?p.toFixed(1):Math.round(p))+'%'}
function fitScale(){const r=views[0].el;const pad=32;const f=Math.min((r.clientWidth-pad*2)/pw,(r.clientHeight-pad*2)/ph);return Math.max(.001,Math.min(f,2/k))}
function render(){
  s=clamp(s,fitScale()*.4,16/k);cx=clamp(cx,0,pw);cy=clamp(cy,0,ph);
  const px=(s*k>=3)?'pixelated':'auto';
  views.forEach(v=>{const W=v.el.clientWidth,H=v.el.clientHeight;const tx=W/2-cx*s,ty=H/2-cy*s;
    v.imgs.forEach(im=>{im.style.transform='translate3d('+tx+'px,'+ty+'px,0) scale('+s+')';im.style.imageRendering=px})});
  zl.textContent=fmtPct(s*k*100);
}
function fit(){s=fitScale();cx=pw/2;cy=ph/2;fitMode=true;render()}
function zoomAt(ns,clientX,clientY){
  ns=clamp(ns,fitScale()*.4,16/k);
  let v=views[0];
  for(const w of views){const r=w.el.getBoundingClientRect();if(clientX>=r.left&&clientX<=r.right&&clientY>=r.top&&clientY<=r.bottom){v=w;break}}
  const r=v.el.getBoundingClientRect(),ax=clientX-r.left,ay=clientY-r.top,W=r.width,H=r.height;
  const ix=cx+(ax-W/2)/s,iy=cy+(ay-H/2)/s;
  s=ns;cx=ix-(ax-W/2)/s;cy=iy-(ay-H/2)/s;fitMode=false;render();
}
function zoomCenter(f){const r=views[0].el.getBoundingClientRect();zoomAt(s*f,r.left+r.width/2,r.top+r.height/2)}
function actual(){const r=views[0].el.getBoundingClientRect();zoomAt(1/k,r.left+r.width/2,r.top+r.height/2)}
function placeDivider(){
  if(!divider)return;const W=views[0].el.clientWidth,x=d*W;
  divider.style.left=x+'px';la.style.clipPath='inset(0 0 0 '+x+'px)';grip.setAttribute('aria-valuenow',Math.round(d*100));
}
function toggleOrig(force){
  if(!altImg)return;showOrig=(typeof force==='boolean')?force:!showOrig;
  altImg.style.visibility=showOrig?'visible':'hidden';
  if(showOrig)setTag(tagEl,cfg.alt.label,cfg.alt.tone);else setTag(tagEl,cfg.panels[0].label,cfg.panels[0].tone);
  bOrig.setAttribute('aria-pressed',showOrig?'true':'false');
}
function reset(){d=.5;placeDivider();toggleOrig(false);fit()}

$('#bIn').onclick=()=>zoomCenter(1.25);
$('#bOut').onclick=()=>zoomCenter(1/1.25);
$('#bFit').onclick=fit;
$('#b100').onclick=actual;
$('#bReset').onclick=reset;
bOrig.onclick=()=>toggleOrig();

stage.addEventListener('wheel',e=>{e.preventDefault();let dy=e.deltaY;if(e.deltaMode===1)dy*=16;zoomAt(s*Math.exp(-dy*.0016),e.clientX,e.clientY)},{passive:false});
stage.addEventListener('dblclick',e=>{if(e.target.closest('.divider'))return;if(Math.abs(s*k-1)<.02)fit();else zoomAt(1/k,e.clientX,e.clientY)});

const ptrs=new Map();let pinch=null;
stage.addEventListener('pointerdown',e=>{
  if(e.target.closest('.divider'))return;
  stage.focus({preventScroll:true});
  try{stage.setPointerCapture(e.pointerId)}catch(_){}
  ptrs.set(e.pointerId,{x:e.clientX,y:e.clientY});stage.classList.add('dragging');
  if(ptrs.size===2){const [a,b]=[...ptrs.values()];pinch={d:Math.hypot(a.x-b.x,a.y-b.y)||1,s:s}}
});
stage.addEventListener('pointermove',e=>{
  const p=ptrs.get(e.pointerId);if(!p)return;
  if(ptrs.size===2&&pinch){p.x=e.clientX;p.y=e.clientY;const [a,b]=[...ptrs.values()];
    zoomAt(pinch.s*Math.hypot(a.x-b.x,a.y-b.y)/pinch.d,(a.x+b.x)/2,(a.y+b.y)/2)}
  else{const dx=e.clientX-p.x,dy=e.clientY-p.y;p.x=e.clientX;p.y=e.clientY;cx-=dx/s;cy-=dy/s;fitMode=false;render()}
});
function endPtr(e){ptrs.delete(e.pointerId);pinch=null;if(!ptrs.size)stage.classList.remove('dragging')}
stage.addEventListener('pointerup',endPtr);stage.addEventListener('pointercancel',endPtr);

if(divider){
  let drag=false;
  const move=e=>{const r=views[0].el.getBoundingClientRect();d=clamp((e.clientX-r.left)/r.width,.02,.98);placeDivider()};
  divider.addEventListener('pointerdown',e=>{drag=true;divider.classList.add('on');try{divider.setPointerCapture(e.pointerId)}catch(_){}move(e);e.preventDefault()});
  divider.addEventListener('pointermove',e=>{if(drag)move(e)});
  const up=()=>{drag=false;divider.classList.remove('on')};
  divider.addEventListener('pointerup',up);divider.addEventListener('pointercancel',up);
  grip.addEventListener('keydown',e=>{
    const step=e.shiftKey?.1:.02;let h=true;
    if(e.key==='ArrowLeft')d=clamp(d-step,.02,.98);
    else if(e.key==='ArrowRight')d=clamp(d+step,.02,.98);
    else if(e.key==='Home')d=.02;else if(e.key==='End')d=.98;else h=false;
    if(h){e.preventDefault();placeDivider()}
  });
}

window.addEventListener('keydown',e=>{
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  const g=document.activeElement;
  if(g&&g.id==='grip'&&['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;
  switch(e.key){
    case '+':case '=':zoomCenter(1.25);break;
    case '-':case '_':zoomCenter(1/1.25);break;
    case '0':fit();break;
    case '1':actual();break;
    case '\\':toggleOrig();break;
    default:return;
  }
  e.preventDefault();
});

new ResizeObserver(()=>{if(fitMode)fit();else render();placeDivider()}).observe(stage);
fit();placeDivider();
})();
</script></body></html>
"""


def build_viewer(config: dict) -> str:
    return VIEWER_HTML.replace("__CONFIG__", json.dumps(config))


# ---------------------------------------------------------
# IMAGE AND METRIC HELPERS (pipeline behaviour unchanged)
# ---------------------------------------------------------

def load_rgb_image(file) -> np.ndarray:
    """Load an uploaded image as a fresh RGB uint8 array."""
    with PILImage.open(file) as image:
        result = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

    if result.size == 0:
        raise ValueError("The uploaded image is empty.")

    return result


def get_metrics(image: np.ndarray) -> dict:
    """Calculate image-quality metrics using the shared API."""
    metrics = calculate_quality_metrics(image)

    return {
        "Mean luminance": metrics.mean_luminance,
        "Luminance contrast": metrics.luminance_contrast,
        "Mean saturation": metrics.mean_saturation,
        "Edge strength": metrics.edge_strength,
        "UIQM": metrics.uiqm,
        "UCIQE": metrics.uciqe,
    }


def encode_png(image: np.ndarray) -> bytes:
    """Encode an RGB image as PNG bytes."""
    buffer = io.BytesIO()
    PILImage.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def process_image(original: np.ndarray) -> dict:
    """Run both enhancement modes and calculate their metrics."""
    start = time.perf_counter()
    fixed = enhance_image(original, adaptive=False)
    fixed_time = time.perf_counter() - start

    start = time.perf_counter()
    adaptive = enhance_image(original, adaptive=True)
    adaptive_time = time.perf_counter() - start

    for name, result in (("Fixed", fixed), ("Adaptive", adaptive)):
        if (
            not isinstance(result, np.ndarray)
            or result.dtype != np.uint8
            or result.shape != original.shape
        ):
            raise ValueError(
                f"{name} enhancement returned an invalid image."
            )

    return {
        "original": original.copy(),
        "fixed": fixed,
        "adaptive": adaptive,
        "fixed_time": fixed_time,
        "adaptive_time": adaptive_time,
        "original_metrics": get_metrics(original),
        "fixed_metrics": get_metrics(fixed),
        "adaptive_metrics": get_metrics(adaptive),
    }


# ---------------------------------------------------------
# PRESENTATION HELPERS
# ---------------------------------------------------------

def build_previews(result: dict) -> dict:
    """Encode display copies for the canvas (exports stay full-resolution)."""
    original = result["original"]
    height, width = original.shape[:2]
    scale = min(1.0, PREVIEW_MAX_EDGE / max(height, width))
    pw = max(1, round(width * scale))
    ph = max(1, round(height * scale))
    resized = (pw, ph) != (width, height)
    use_png = pw * ph <= PREVIEW_PNG_PIXELS

    def encode(array: np.ndarray) -> str:
        image = PILImage.fromarray(array)
        if resized:
            image = image.resize((pw, ph), RESAMPLE)
        buffer = io.BytesIO()
        if use_png:
            image.save(buffer, format="PNG", compress_level=3)
            mime = "image/png"
        else:
            image.save(buffer, format="JPEG", quality=92, subsampling=0)
            mime = "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")

    def thumb(array: np.ndarray) -> str:
        image = PILImage.fromarray(array)
        image.thumbnail((640, 640), RESAMPLE)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")

    return {
        "v": PREVIEW_VERSION,
        "original": encode(original),
        "fixed": encode(result["fixed"]),
        "adaptive": encode(result["adaptive"]),
        "th_original": thumb(original),
        "th_fixed": thumb(result["fixed"]),
        "th_adaptive": thumb(result["adaptive"]),
        "pw": pw,
        "ph": ph,
        "ow": width,
        "oh": height,
        "scaled": resized,
    }


def histogram_image(image: np.ndarray) -> str:
    """Return an <img> tag holding an additive RGB histogram (SVG)."""
    step = max(1, int(np.sqrt(image.shape[0] * image.shape[1] / 250_000)))
    sample = image[::step, ::step]
    bins, width, height = 64, 256, 90

    counts = [
        np.histogram(sample[..., c], bins=bins, range=(0, 256))[0].astype(float)
        for c in range(3)
    ]
    reference = max(float(np.percentile(np.concatenate(counts), 99.5)), 1.0)

    shapes = []
    for values, colour in zip(counts, ("#ff4d57", "#35d47f", "#4d8dff")):
        ys = height - np.clip(values / reference, 0, 1) * (height - 6)
        points = [f"0,{height}"]
        points += [
            f"{(i + 0.5) * width / bins:.1f},{y:.1f}" for i, y in enumerate(ys)
        ]
        points.append(f"{width},{height}")
        shapes.append(
            f'<polygon points="{" ".join(points)}" fill="{colour}" '
            f'fill-opacity=".62" style="mix-blend-mode:screen"/>'
        )

    grid = "".join(
        f'<line x1="{x}" y1="0" x2="{x}" y2="{height}" '
        f'stroke="#ffffff" stroke-opacity=".08" stroke-width="1"/>'
        for x in (64, 128, 192)
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} '
        f'{height}" preserveAspectRatio="none"><rect width="{width}" '
        f'height="{height}" fill="#26262d"/>{grid}{"".join(shapes)}</svg>'
    )
    return f'<img class="hist" alt="RGB histogram" src="{svg_uri(svg)}">'


def human_size(num_bytes: float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB"):
        if value < 1024:
            return f"{value:.0f} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def metric_valid(value) -> bool:
    try:
        return value is not None and bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def fmt_value(value) -> str:
    if not metric_valid(value):
        return "—"
    value = float(value)
    return f"{value:.3f}" if abs(value) < 10 else f"{value:,.2f}"


def fmt_delta(value, reference) -> str:
    if not (metric_valid(value) and metric_valid(reference)):
        return "—"
    reference = float(reference)
    if reference == 0:
        return "—"
    change = (float(value) - reference) / abs(reference) * 100
    return f"{change:+.1f}% vs original"


def available_metrics(result: dict) -> list:
    names = list(CORE_METRICS)
    for extra in EXTRA_METRICS:
        if any(
            metric_valid(result[f"{src}_metrics"].get(extra))
            for src in ("original", "fixed", "adaptive")
        ):
            names.append(extra)
    return names


def get_png(result: dict, key: str) -> bytes:
    """Encode a full-resolution PNG once and reuse it."""
    cache = st.session_state.setdefault("aq_exports", {})
    if key not in cache:
        cache[key] = encode_png(result[key])
    return cache[key]


# ---------------------------------------------------------
# STATE CALLBACKS
# ---------------------------------------------------------

def reset_workspace() -> None:
    """Clear the loaded image and cached results; show a fresh uploader."""
    for key in (
        "aq_result", "aq_previews", "aq_name", "aq_size",
        "aq_exports", "aq_hist", "aq_err",
    ):
        st.session_state.pop(key, None)
    st.session_state["aq_gen"] = st.session_state.get("aq_gen", 0) + 1


def open_tool(tool_id: str) -> None:
    st.session_state["aq_nav"] = TOOLS[tool_id]["nav"]


def go_home() -> None:
    st.session_state["aq_nav"] = "Home"


# ---------------------------------------------------------
# CHROME
# ---------------------------------------------------------

def render_topbar() -> str:
    brand, nav, right = st.columns([1, 1, 1])

    with brand:
        md(
            f"""
            <div class="aq-brand">
                <img src="{LOGO}" alt="" width="36" height="36">
                <span class="bn">Aqua<b>Restore</b></span>
            </div>
            """
        )
        st.button("Go to home page", key="aq_home", on_click=go_home)

    if st.session_state.get("aq_nav") not in NAV:
        st.session_state["aq_nav"] = "Home"

    with nav:
        choice = st.radio(
            "Tools",
            NAV,
            horizontal=True,
            key="aq_nav",
            label_visibility="collapsed",
        )

    with right:
        if "aq_result" in st.session_state:
            st.button(
                "New image",
                key="aq_new",
                type="primary",
                on_click=reset_workspace,
            )
        else:
            md('<div class="tb-r"><span class="dot"></span>Local processing</div>')

    return choice


def hero(
    title_html: str,
    subtitle: str,
    chip: str = "",
    tight: bool = False,
    eyebrow: str = "",
) -> None:
    """Page headline. ``title_html`` and ``eyebrow`` are trusted constants."""
    cls = "hero tight" if tight else "hero"
    eyebrow_html = f'<div class="eb">{escape(eyebrow)}</div>' if eyebrow else ""
    chip_html = (
        f'<div class="hc"><span class="dot"></span>{escape(chip)}</div>'
        if chip
        else ""
    )
    md(
        f"""
        <div class="{cls}">
            {eyebrow_html}
            <div class="ht">{title_html}</div>
            <div class="hs">{escape(subtitle)}</div>
            {chip_html}
        </div>
        """
    )


def footer() -> None:
    md(
        """
        <div class="aq-foot">
            <span>&copy; AquaRestore 2026 &middot; Underwater image restoration</span>
            <span>Classical image processing &middot; RGB &middot; 8-bit &middot; processed on this device</span>
        </div>
        """
    )


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------

def render_home() -> None:
    chip = ""
    if "aq_result" in st.session_state:
        previews = st.session_state["aq_previews"]
        chip = (
            f"{st.session_state['aq_name']} · "
            f"{previews['ow']} × {previews['oh']} px loaded"
        )

    hero(
        "Every tool you need to <em>restore</em> underwater images in one place",
        "Every tool you need to bring colour and clarity back, at your "
        "fingertips. Enhance, compare, measure and export underwater "
        "photographs with just a few clicks, all processed on your device.",
        chip=chip,
        eyebrow="Underwater image restoration",
    )

    md('<div class="pill-a"></div>')
    selected = st.radio(
        "Filter tools",
        FILTERS,
        horizontal=True,
        key="aq_filter",
        label_visibility="collapsed",
    )
    st.markdown(pill_css(selected), unsafe_allow_html=True)

    visible = [
        tid for tid, tool in TOOLS.items()
        if selected == "All" or tool["cat"] == selected
    ]

    per_row = 4
    for start in range(0, len(visible), per_row):
        columns = st.columns(per_row, gap="medium")
        for column, tool_id in zip(columns, visible[start:start + per_row]):
            tool = TOOLS[tool_id]
            with column:
                md(
                    f"""
                    <div class="tool-card">
                        <span class="tc">{escape(tool['cat'])}</span>
                        <img class="ti" alt="" src="{tool_icon(tool_id)}">
                        <div class="tt">{escape(tool['title'])}</div>
                        <div class="td">{escape(tool['desc'])}</div>
                    </div>
                    """
                )
                st.button(
                    f"Open {tool['title']}",
                    key=f"aq_card_{tool_id}",
                    on_click=open_tool,
                    args=(tool_id,),
                )

    footer()


# ---------------------------------------------------------
# TOOL LANDING (select image)
# ---------------------------------------------------------

def render_landing(tool: dict):
    landing = st.empty()
    with landing.container():
        hero(tool["title_html"], tool["subtitle"], tight=True, eyebrow=tool["cat"])
        md('<div style="height:8px"></div>')
        uploaded = st.file_uploader(
            "Select an image",
            type=UPLOAD_TYPES,
            label_visibility="collapsed",
            key=f"aq_upload_{st.session_state.get('aq_gen', 0)}",
        )
        md(
            '<div class="upnote">PNG, JPEG, BMP or TIFF &middot; up to 200 MB '
            "&middot; processed locally and never uploaded</div>"
        )
    return landing, uploaded


def render_processing() -> None:
    md(
        """
        <div class="proc">
            <div class="pt">Processing image</div>
            <div class="ps">Running fixed and adaptive enhancement and measuring quality.</div>
            <div class="lbar"><i></i></div>
        </div>
        """
    )


def render_error() -> None:
    md(
        f"""
        <div class="err">
            <div class="et">This image could not be processed</div>
            <div class="em">{escape(st.session_state.get('aq_err', 'Unknown error.'))}</div>
            <div class="eh">Confirm the file is a valid PNG, JPEG, BMP or TIFF image, then try again with another file.</div>
        </div>
        """
    )
    _left, middle, _right = st.columns([1, 1, 1])
    with middle:
        st.button(
            "Choose another image",
            key="aq_err_reset",
            type="primary",
            on_click=reset_workspace,
        )


# ---------------------------------------------------------
# WORKSPACE (canvas + options panel)
# ---------------------------------------------------------

TONE_OF = {"original": "orig", "fixed": "fixed", "adaptive": "adaptive"}
VAR_OF = {"original": "--orig", "fixed": "--fixed", "adaptive": "--adaptive"}


def kv(key: str, value: str) -> str:
    return f'<div class="kv"><span class="k">{key}</span><span class="v">{value}</span></div>'


def viewer_config(tool_id: str, mode: str, layout: str, prev: dict) -> dict:
    key = mode.lower()
    base = {"pw": prev["pw"], "ph": prev["ph"], "ow": prev["ow"]}
    original = {"label": "Original", "src": prev["original"], "tone": "orig"}
    fixed = {"label": "Fixed", "src": prev["fixed"], "tone": "fixed"}
    adaptive = {"label": "Adaptive", "src": prev["adaptive"], "tone": "adaptive"}
    selected = {"label": mode, "src": prev[key], "tone": MODE_TONE[mode]}

    if tool_id in ("fixed", "adaptive"):
        return {
            **base, "layout": "single", "tone": MODE_TONE[mode],
            "title": f"{mode} enhancement", "sub": "Single view",
            "panels": [{**selected, "label": f"{mode} enhancement"}],
            "alt": original,
        }
    if tool_id == "beforeafter":
        return {
            **base, "layout": "slider", "tone": MODE_TONE[mode],
            "title": f"{mode} enhancement", "sub": "Before / after",
            "panels": [original, selected],
        }
    if layout == "Original and enhanced":
        return {
            **base, "layout": "split", "tone": MODE_TONE[mode],
            "title": f"Original and {mode.lower()}", "sub": "Side by side",
            "panels": [original, selected],
        }
    if layout == "Fixed and adaptive":
        return {
            **base, "layout": "split", "tone": "ink",
            "title": "Fixed and adaptive", "sub": "Enhancement comparison",
            "panels": [fixed, adaptive],
        }
    return {
        **base, "layout": "split", "tone": "ink",
        "title": "Original, fixed and adaptive", "sub": "All three",
        "panels": [original, fixed, adaptive],
    }


def panel_sections(name, size, previews, result, key, histogram) -> str:
    width, height = previews["ow"], previews["oh"]
    megapixels = width * height / 1_000_000
    extension = name.rsplit(".", 1)[-1].upper() if "." in name else "—"
    preview_text = (
        f'{previews["pw"]} × {previews["ph"]} (scaled)'
        if previews["scaled"]
        else "Native"
    )

    image_block = (
        f'<div class="fname" title="{escape(name)}">{escape(name)}</div>'
        + kv("Dimensions", f"{width} × {height} px")
        + kv("Megapixels", f"{megapixels:.2f} MP")
        + kv("Source", f"{extension} · {human_size(size)}")
        + kv("Colour", "RGB · 8-bit")
        + kv("Canvas preview", preview_text)
    )

    blocks = []
    for metric in available_metrics(result):
        raw = {
            source: result[f"{source}_metrics"].get(metric)
            for source in ("original", "fixed", "adaptive")
        }
        peak = max(
            [abs(float(v)) for v in raw.values() if metric_valid(v)] or [1.0]
        ) or 1.0
        cells = []
        for label, source in (
            ("Original", "original"),
            ("Fixed", "fixed"),
            ("Adaptive", "adaptive"),
        ):
            value = raw[source]
            bar = abs(float(value)) / peak * 100 if metric_valid(value) else 0
            cells.append(
                f'<div class="c"><span class="cl">{label}</span>'
                f'<span class="cv">{fmt_value(value)}</span>'
                f'<span class="cb"><i style="width:{bar:.1f}%;'
                f'background:var({VAR_OF[source]})"></i></span></div>'
            )
        blocks.append(
            f'<div class="mb"><div class="mn">{metric}</div>'
            f'<div class="mv">{"".join(cells)}</div></div>'
        )

    processing = (
        kv("Fixed", f"{result['fixed_time'] * 1000:.2f} ms")
        + kv("Adaptive", f"{result['adaptive_time'] * 1000:.2f} ms")
        + kv("Status", "Complete")
        + '<div class="note" style="margin-top:10px">Timings measure '
        "enhancement only and vary with image size and system load.</div>"
    )

    tone = TONE_OF[key]
    shown = key.capitalize()
    return f"""
    <div class="psec first"><div class="ih"><span>Image</span></div>{image_block}</div>
    <div class="psec">
        <div class="ih"><span>Histogram</span><span class="hm"><i class="tdot t-{tone}"></i>{escape(shown)}</span></div>
        {histogram}
        <div class="hax"><span>Shadows</span><span>Midtones</span><span>Highlights</span></div>
    </div>
    <div class="psec">
        <div class="ih"><span>Quality metrics</span></div>
        <div class="mlist">{"".join(blocks)}</div>
    </div>
    <div class="psec"><div class="ih"><span>Processing</span></div>{processing}</div>
    """


def render_workspace(tool_id, tool, name, size, previews, result) -> None:
    state = st.session_state
    viewer_col, panel_col = st.columns([1, 1])

    mode = "Fixed" if tool_id == "fixed" else "Adaptive" if tool_id == "adaptive" else None
    layout = COMPARE_LAYOUTS[0]
    needs_mode = tool_id == "beforeafter"

    with panel_col:
        md(
            f"""
            <div class="panel-head">
                <div class="pe">{escape(tool['cat'])}</div>
                <div class="pk">{tool['title_html']}</div>
                <div class="pdsc">{escape(tool['subtitle'])}</div>
            </div>
            """
        )

        if tool_id == "compare":
            md('<div class="plabel">Layout</div>')
            md('<div class="lst-a"></div>')
            layout = st.radio(
                "Compare layout",
                COMPARE_LAYOUTS,
                key="aq_cmp",
                label_visibility="collapsed",
            )
            st.markdown(list_css(layout), unsafe_allow_html=True)
            needs_mode = layout == "Original and enhanced"

        if needs_mode:
            md('<div class="plabel">Enhancement</div>')
            md('<div class="seg-a"></div>')
            mode = st.radio(
                "Enhancement method",
                MODES,
                index=1,
                horizontal=True,
                key="aq_mode",
                label_visibility="collapsed",
            )
            st.markdown(seg_css(mode), unsafe_allow_html=True)
            md(f'<div class="pnote">{escape(MODE_NOTES[mode])}</div>')

        if mode is None:
            mode = state.get("aq_mode", "Adaptive")

    key = mode.lower()
    hist_cache = state.setdefault("aq_hist", {})
    if key not in hist_cache:
        hist_cache[key] = histogram_image(result[key])

    with viewer_col:
        components.html(
            build_viewer(viewer_config(tool_id, mode, layout, previews)),
            height=720,
            scrolling=False,
        )

    with panel_col:
        md(panel_sections(name, size, previews, result, key, hist_cache[key]))

        md('<div class="cta-a"></div>')
        if tool_id == "compare":
            st.button(
                "Continue to export",
                key="aq_cta_export",
                type="primary",
                on_click=open_tool,
                args=("export",),
            )
        else:
            with st.spinner("Preparing PNG..."):
                data = get_png(result, key)
            st.download_button(
                f"Download {key} PNG",
                data=data,
                file_name=f"aquarestore_{key}.png",
                mime="image/png",
                key=f"aq_cta_{tool_id}_{key}",
                type="primary",
            )


# ---------------------------------------------------------
# RESULT PAGES
# ---------------------------------------------------------

def render_analysis_page(tool, name, previews, result) -> None:
    hero(tool["title_html"], tool["subtitle"], chip=name, tight=True, eyebrow=tool["cat"])

    names = available_metrics(result)
    fixed_ms = result["fixed_time"] * 1000
    adaptive_ms = result["adaptive_time"] * 1000
    megapixels = previews["ow"] * previews["oh"] / 1_000_000

    tiles = (
        ("Resolution", f'{previews["ow"]} × {previews["oh"]}', f"{megapixels:.2f} megapixels", "orig"),
        ("Fixed runtime", f"{fixed_ms:.1f} ms", "enhancement only", "fixed"),
        ("Adaptive runtime", f"{adaptive_ms:.1f} ms", "enhancement only", "adaptive"),
        ("Metrics tracked", str(len(names)), "per image", "ink"),
    )
    md(
        '<div class="kpis">'
        + "".join(
            f'<div class="kpi k-{tone}"><div class="kl">{label}</div>'
            f'<div class="kn">{value}</div><div class="ks">{sub}</div></div>'
            for label, value, sub, tone in tiles
        )
        + "</div>"
    )

    header = (
        '<div class="gr gh"><div>Metric</div>'
        '<div><i class="tdot t-orig"></i>Original</div>'
        '<div><i class="tdot t-fixed"></i>Fixed</div>'
        '<div><i class="tdot t-adaptive"></i>Adaptive</div></div>'
    )

    rows = []
    for metric in names:
        original = result["original_metrics"].get(metric)
        values = {
            "o": original,
            "f": result["fixed_metrics"].get(metric),
            "a": result["adaptive_metrics"].get(metric),
        }
        peak = max(
            [abs(float(v)) for v in values.values() if metric_valid(v)] or [1.0]
        ) or 1.0

        cells = []
        for tone, value in values.items():
            width = abs(float(value)) / peak * 100 if metric_valid(value) else 0
            delta = "Reference" if tone == "o" else fmt_delta(value, original)
            cells.append(
                f'<div><div class="av">{fmt_value(value)}</div>'
                f'<div class="ad">{delta}</div>'
                f'<div class="ab"><i class="{tone}" style="width:{width:.1f}%"></i></div></div>'
            )
        rows.append(
            f'<div class="gr"><div class="mname">{metric}</div>{"".join(cells)}</div>'
        )

    md(f'<div class="card gt">{header}{"".join(rows)}</div>')

    peak_time = max(fixed_ms, adaptive_ms, 1e-9)
    md(
        f"""
        <div class="ag">
            <div class="card pc">
                <div class="ih"><span>Processing performance</span></div>
                <div class="rt"><span><i class="tdot t-fixed"></i>Fixed</span><span class="rv">{fixed_ms:.2f} ms</span></div>
                <div class="ab"><i class="f" style="width:{fixed_ms / peak_time * 100:.1f}%"></i></div>
                <div class="rt" style="margin-top:16px"><span><i class="tdot t-adaptive"></i>Adaptive</span><span class="rv">{adaptive_ms:.2f} ms</span></div>
                <div class="ab"><i class="a" style="width:{adaptive_ms / peak_time * 100:.1f}%"></i></div>
                <div class="note">Times measure enhancement execution only and vary by image size and system load.</div>
            </div>
            <div class="card pc">
                <div class="ih"><span>Reading these metrics</span></div>
                <div class="note" style="margin-top:0;font-size:15px">Metrics describe image properties. Higher luminance, contrast, saturation or edge strength is not universally better. UIQM and UCIQE appear only when verified implementations are integrated.</div>
            </div>
        </div>
        """
    )
    footer()


def render_export_page(tool, name, previews, result) -> None:
    hero(tool["title_html"], tool["subtitle"], chip=name, tight=True, eyebrow=tool["cat"])

    cache = st.session_state.setdefault("aq_exports", {})
    missing = [k for k in ("fixed", "adaptive", "original") if k not in cache]
    if missing:
        with st.spinner("Preparing lossless PNG files..."):
            for key in missing:
                cache[key] = encode_png(result[key])

    active = st.session_state.get("aq_mode", "Adaptive").lower()
    width, height = previews["ow"], previews["oh"]
    items = (
        ("fixed", "Fixed enhancement", "Output of the fixed enhancement pipeline.", "aquarestore_fixed.png"),
        ("adaptive", "Adaptive enhancement", "Output of the adaptive enhancement pipeline.", "aquarestore_adaptive.png"),
        ("original", "Original", "The imported image, re-encoded losslessly as PNG.", "aquarestore_original.png"),
    )

    md('<div style="height:8px"></div>')
    columns = st.columns(3, gap="medium")
    for column, (key, title, description, filename) in zip(columns, items):
        selected = key == active
        badge = '<span class="xb">Recommended</span>' if key == "adaptive" else ""
        with column:
            md(
                f"""
                <div class="card xc">
                    <div class="xth" style="--t:var({VAR_OF[key]})"><img alt="{escape(title)} preview" src="{previews['th_' + key]}"></div>
                    <div class="xt"><span><i class="tdot t-{TONE_OF[key]}"></i>{title}</span>{badge}</div>
                    <div class="xd">{description}</div>
                    {kv("Format", "PNG · lossless")}
                    {kv("Dimensions", f"{width} × {height} px")}
                    {kv("File size", human_size(len(cache[key])))}
                </div>
                """
            )
            st.download_button(
                f"Download {title.lower()}",
                data=cache[key],
                file_name=filename,
                mime="image/png",
                key=f"aq_dl_{key}",
                type="primary" if selected else "secondary",
            )
    footer()


# ---------------------------------------------------------
# APPLICATION
# ---------------------------------------------------------

st.markdown(f"<style>{_compact(CSS)}</style>", unsafe_allow_html=True)

nav = render_topbar()
tool_id = NAV_TO_TOOL[nav]

st.markdown(nav_css(nav), unsafe_allow_html=True)

if tool_id is None:
    st.markdown(f"<style>{CENTERED_CSS}</style>", unsafe_allow_html=True)
    render_home()
    st.stop()

tool = TOOLS[tool_id]
state = st.session_state

if "aq_result" not in state:
    st.markdown(f"<style>{CENTERED_CSS}</style>", unsafe_allow_html=True)

    if state.get("aq_err"):
        render_error()
        st.stop()

    landing, uploaded_file = render_landing(tool)
    if uploaded_file is None:
        footer()
        st.stop()

    landing.empty()
    loading = st.empty()
    with loading.container():
        render_processing()

    try:
        original = load_rgb_image(uploaded_file)
        result = process_image(original)
        previews = build_previews(result)
    except Exception as exc:
        state["aq_err"] = f"{type(exc).__name__}: {exc}"
        loading.empty()
        st.rerun()

    state["aq_result"] = result
    state["aq_previews"] = previews
    state["aq_name"] = uploaded_file.name
    state["aq_size"] = uploaded_file.size
    state["aq_exports"] = {}
    state["aq_hist"] = {}
    loading.empty()
    st.rerun()

result = state["aq_result"]
name = state["aq_name"]
size = state["aq_size"]

if state["aq_previews"].get("v") != PREVIEW_VERSION:
    state["aq_previews"] = build_previews(result)
    state["aq_hist"] = {}
previews = state["aq_previews"]

if tool_id in ("analysis", "export"):
    st.markdown(f"<style>{CENTERED_CSS}</style>", unsafe_allow_html=True)
    if tool_id == "analysis":
        render_analysis_page(tool, name, previews, result)
    else:
        render_export_page(tool, name, previews, result)
else:
    render_workspace(tool_id, tool, name, size, previews, result)