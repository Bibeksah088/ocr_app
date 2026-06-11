import streamlit as st
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import io
import threading
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.ocr_engine import OCREngine
from modules.preprocessor import ImagePreprocessor
from modules.exporter import Exporter
import config

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCR Research Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background: #07070f;
    font-family: 'Syne', sans-serif;
}
[data-testid="stSidebar"] {
    background: #0c0c1a !important;
    border-right: 1px solid #1a1a30;
}
[data-testid="stSidebar"] * { font-family: 'Syne', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0d0d22 0%, #10102e 40%, #0a1628 100%);
    border: 1px solid #1e1e40;
    border-radius: 20px;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    text-align: center;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(124,58,237,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; right: -40px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(37,99,235,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    line-height: 1.1;
}
.hero-sub {
    color: #6b7280;
    font-size: 0.95rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.badge-row { display: flex; gap: 0.5rem; flex-wrap: wrap; justify-content: center; }
.badge {
    padding: 0.25rem 0.9rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
}
.b-v { background: rgba(124,58,237,0.15); color: #a78bfa; border: 1px solid rgba(124,58,237,0.3); }
.b-b { background: rgba(37,99,235,0.15);  color: #60a5fa; border: 1px solid rgba(37,99,235,0.3);  }
.b-g { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }

.metric-card {
    background: #0c0c1a;
    border: 1px solid #1a1a30;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
}
.metric-card:hover { border-color: #3d3d6b; transform: translateY(-2px); }
.mv { font-family: 'Space Mono', monospace; font-size: 1.7rem; font-weight: 700; }
.ml { font-size: 0.68rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem; }
.c-hi { color: #34d399; }
.c-md { color: #fbbf24; }
.c-lo { color: #f87171; }

.sec-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1a1a30;
    font-family: 'Space Mono', monospace;
}
.result-box {
    background: #090914;
    border: 1px solid #1a1a30;
    border-radius: 12px;
    padding: 1.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #d1d5db;
    white-space: pre-wrap;
    line-height: 1.7;
    max-height: 380px;
    overflow-y: auto;
}
.result-box::-webkit-scrollbar { width: 4px; }
.result-box::-webkit-scrollbar-track { background: #0c0c1a; }
.result-box::-webkit-scrollbar-thumb { background: #3d3d6b; border-radius: 2px; }

.info-box {
    background: #0c0c1a;
    border: 1px solid #1a1a30;
    border-left: 3px solid #a78bfa;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem;
    color: #9ca3af;
    font-size: 0.83rem;
    margin-bottom: 1rem;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.25s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.4) !important;
}
.stDownloadButton > button {
    background: #0c0c1a !important;
    color: #a78bfa !important;
    border: 1px solid #3d3d6b !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
.stDownloadButton > button:hover {
    background: #1a1a30 !important;
    border-color: #a78bfa !important;
}
div[data-testid="stTabs"] > div > div > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
    font-size: 0.85rem !important;
}
div[data-testid="stTabs"] > div > div > button[aria-selected="true"] {
    color: #a78bfa !important;
}
div[data-baseweb="select"] { background: #0c0c1a !important; }
.stSlider > div > div > div { background: linear-gradient(90deg, #7c3aed, #2563eb) !important; }
label { color: #9ca3af !important; font-family: 'Syne', sans-serif !important; font-size: 0.82rem !important; }
.stRadio label { color: #d1d5db !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def pil_to_bytes(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def conf_cls(c):
    return "c-hi" if c >= 80 else ("c-md" if c >= 60 else "c-lo")


def metric_card(val, label, extra_cls=""):
    return f'<div class="metric-card"><div class="mv {extra_cls}">{val}</div><div class="ml">{label}</div></div>'


def meta_for(lang_name, preprocess_mode, conf, words):
    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "confidence": f"{conf:.1f}",
        "lang": lang_name,
        "preprocess": preprocess_mode,
        "words": words,
    }


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {"history": [], "last_text": "", "last_conf": 0.0}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:Syne,sans-serif;font-size:1.2rem;font-weight:800;color:#a78bfa;margin:0 0 1.5rem">⚙ Configuration</p>', unsafe_allow_html=True)

    lang_name = st.selectbox("🌐 Language", list(config.LANGS.keys()))
    lang = config.LANGS[lang_name]

    preprocess_mode = st.selectbox("🔧 Preprocessing Mode", config.PREPROCESS_MODES, index=8)

    psm_name = st.selectbox("📄 Page Segmentation (PSM)", list(config.PSM_MODES.keys()))
    psm = config.PSM_MODES[psm_name]

    oem_name = st.selectbox("🤖 OCR Engine (OEM)", list(config.OEM_MODES.keys()), index=3)
    oem = config.OEM_MODES[oem_name]

    st.divider()

    conf_thresh = st.slider("🎯 Box Confidence Threshold", 0, 100, 60, 5)
    show_boxes = st.toggle("Draw Bounding Boxes", value=True)

    st.divider()

    total = len(st.session_state["history"])
    avg_c = round(sum(r["confidence"] for r in st.session_state["history"]) / total, 1) if total else 0
    st.markdown(f"""
    <div style="background:#090914;border:1px solid #1a1a30;border-radius:10px;padding:1rem;font-family:'Space Mono',monospace;font-size:0.75rem;color:#6b7280">
        <span style="color:#a78bfa;font-weight:700">OCR Research Lab</span><br>
        v{config.VERSION} · Tesseract + OpenCV<br><br>
        Runs this session: <span style="color:#60a5fa">{total}</span><br>
        Avg confidence:    <span style="color:#34d399">{avg_c}%</span>
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🔬 OCR Research Lab</div>
    <div class="hero-sub">Optical Character Recognition · Tesseract 5 Engine · OpenCV Preprocessing Pipeline</div>
    <div class="badge-row">
        <span class="badge b-v">Tesseract 5.x</span>
        <span class="badge b-b">OpenCV 4.x</span>
        <span class="badge b-g">Real-time Camera</span>
        <span class="badge b-v">9 Preprocess Modes</span>
        <span class="badge b-b">Multi-language</span>
        <span class="badge b-g">Batch Processing</span>
        <span class="badge b-v">PDF Export</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "📷  Image Upload",
    "🎥  Live Camera",
    "📦  Batch Process",
    "🔍  Compare Modes",
    "📊  Analytics",
    "🎙️  Speech to Text",
    
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — IMAGE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
with t1:
    st.markdown('<div class="sec-title">Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag & drop or browse — JPG, PNG, BMP, TIFF, WEBP",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        raw = Image.open(uploaded).convert("RGB")

        with st.spinner("Running OCR pipeline…"):
            proc = ImagePreprocessor.process(raw, preprocess_mode)
            data, elapsed = OCREngine.extract_with_data(proc, lang, psm, oem)
            text, _ = OCREngine.extract_text(proc, lang, psm, oem)
            conf = OCREngine.get_confidence(data)
            stats = OCREngine.text_stats(text)
            ann = OCREngine.draw_boxes(proc, data, conf_thresh) if show_boxes else proc

        st.session_state["history"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": uploaded.name,
            "language": lang_name,
            "preprocessing": preprocess_mode,
            "confidence": round(conf, 2),
            "words": stats["words"],
            "characters": stats["characters"],
            "time_ms": round(elapsed * 1000, 1),
        })

        # Metrics
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.markdown(metric_card(f"{conf:.1f}%", "Confidence", conf_cls(conf)), unsafe_allow_html=True)
        with c2: st.markdown(metric_card(stats["words"], "Words"), unsafe_allow_html=True)
        with c3: st.markdown(metric_card(stats["characters"], "Characters"), unsafe_allow_html=True)
        with c4: st.markdown(metric_card(stats["lines"], "Lines"), unsafe_allow_html=True)
        with c5: st.markdown(metric_card(f"{elapsed*1000:.0f}ms", "Process Time"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Images
        ic1, ic2 = st.columns(2)
        with ic1:
            st.markdown('<div class="sec-title">Original</div>', unsafe_allow_html=True)
            st.image(raw, use_column_width=True)
        with ic2:
            lbl = "Processed + Bounding Boxes" if show_boxes else f"Processed — {preprocess_mode}"
            st.markdown(f'<div class="sec-title">{lbl}</div>', unsafe_allow_html=True)
            st.image(ann, use_column_width=True)

        # Text output
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Extracted Text</div>', unsafe_allow_html=True)
        if text:
            st.markdown(f'<div class="result-box">{text}</div>', unsafe_allow_html=True)
        else:
            st.warning("No text detected. Try changing the preprocessing mode or PSM setting.")

        # Word confidence table
      
              # Word confidence table
        words = OCREngine.get_word_stats(data, conf_thresh)
        if words:
            with st.expander("📋 Word-level Confidence Analysis"):
                df_w = pd.DataFrame(words).sort_values(
                    "confidence",
                    ascending=False
                )

                st.dataframe(
                    df_w,
                    use_container_width=True,
                    height=260,
                )
        # Export
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Export</div>', unsafe_allow_html=True)
        meta = meta_for(lang_name, preprocess_mode, conf, stats["words"])
        e1, e2, e3 = st.columns(3)
        with e1:
            st.download_button("📄 Download TXT", Exporter.to_txt(text, meta), "ocr_result.txt", "text/plain", use_container_width=True)
        with e2:
            pdf = Exporter.to_pdf_bytes(text, meta)
            if pdf:
                st.download_button("📑 Download PDF", pdf, "ocr_result.pdf", "application/pdf", use_container_width=True)
            else:
                st.info("Install fpdf2 for PDF export")
        with e3:
            st.download_button("🖼️ Annotated Image", pil_to_bytes(ann), "annotated.png", "image/png", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — LIVE CAMERA
# ─────────────────────────────────────────────────────────────────────────────
with t2:
    cam_mode = st.radio("Mode", ["📸 Snapshot (Stable)", "🎥 Live Stream (WebRTC)"], horizontal=True)

    if cam_mode == "📸 Snapshot (Stable)":
        st.markdown('<div class="info-box">📸 Capture a photo — OCR runs automatically on the snapshot.</div>', unsafe_allow_html=True)
        cam_img = st.camera_input("", label_visibility="collapsed")

        if cam_img:
            raw_c = Image.open(cam_img).convert("RGB")
            with st.spinner("Running OCR…"):
                proc_c = ImagePreprocessor.process(raw_c, preprocess_mode)
                data_c, elapsed_c = OCREngine.extract_with_data(proc_c, lang, psm, oem)
                text_c, _ = OCREngine.extract_text(proc_c, lang, psm, oem)
                conf_c = OCREngine.get_confidence(data_c)
                ann_c = OCREngine.draw_boxes(proc_c, data_c, conf_thresh) if show_boxes else proc_c

            st.session_state["history"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "Camera Snapshot",
                "language": lang_name,
                "preprocessing": preprocess_mode,
                "confidence": round(conf_c, 2),
                "words": len(text_c.split()),
                "characters": len(text_c),
                "time_ms": round(elapsed_c * 1000, 1),
            })

            m1, m2, m3 = st.columns(3)
            with m1: st.markdown(metric_card(f"{conf_c:.1f}%", "Confidence", conf_cls(conf_c)), unsafe_allow_html=True)
            with m2: st.markdown(metric_card(len(text_c.split()), "Words"), unsafe_allow_html=True)
            with m3: st.markdown(metric_card(f"{elapsed_c*1000:.0f}ms", "Time"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            ci1, ci2 = st.columns(2)
            with ci1:
                st.markdown('<div class="sec-title">Captured</div>', unsafe_allow_html=True)
                st.image(raw_c, use_column_width=True)
            with ci2:
                st.markdown('<div class="sec-title">Annotated</div>', unsafe_allow_html=True)
                st.image(ann_c, use_column_width=True)

            st.markdown('<div class="sec-title">Extracted Text</div>', unsafe_allow_html=True)
            disp = text_c if text_c else "(No text detected)"
            st.markdown(f'<div class="result-box">{disp}</div>', unsafe_allow_html=True)

            meta_c = meta_for(lang_name, preprocess_mode, conf_c, len(text_c.split()))
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button("📄 TXT", Exporter.to_txt(text_c, meta_c), "camera_ocr.txt", "text/plain", use_container_width=True)
            with dl2:
                pdf_c = Exporter.to_pdf_bytes(text_c, meta_c)
                if pdf_c:
                    st.download_button("📑 PDF", pdf_c, "camera_ocr.pdf", "application/pdf", use_container_width=True)

    else:
        st.markdown('<div class="info-box">🎥 Live OCR — bounding boxes drawn in real-time. Requires camera permission.</div>', unsafe_allow_html=True)
        try:
            from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
            import av
            import pytesseract

            RTC = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

            skip_n = st.slider("Process every N frames (higher = lighter CPU)", 10, 60, 25, 5)

            class LiveOCR(VideoProcessorBase):
                def __init__(self):
                    self.cnt = 0
                    self.lock = threading.Lock()
                    self.txt = ""
                    self.cf = 0.0
                    self.last_data = None

                def recv(self, frame):
                    bgr = frame.to_ndarray(format="bgr24")
                    self.cnt += 1
                    if self.cnt % skip_n == 0:
                        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                        proc_l = ImagePreprocessor.process(pil, preprocess_mode)
                        d, _ = OCREngine.extract_with_data(proc_l, lang, psm, oem)
                        t = pytesseract.image_to_string(proc_l, lang=lang, config=f"--psm {psm} --oem {oem}").strip()
                        c = OCREngine.get_confidence(d)
                        with self.lock:
                            self.txt = t
                            self.cf = c
                            self.last_data = d
                        for i in range(len(d["level"])):
                            cf_i = int(d["conf"][i])
                            if cf_i < conf_thresh or cf_i == -1:
                                continue
                            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
                            if w > 0 and h > 0:
                                col = (0, 220, 120) if cf_i >= 80 else ((0, 165, 255) if cf_i >= 60 else (0, 80, 255))
                                cv2.rectangle(bgr, (x, y), (x + w, y + h), col, 2)
                    return av.VideoFrame.from_ndarray(bgr, format="bgr24")

            vc1, vc2 = st.columns([2, 1])
            with vc1:
                ctx = webrtc_streamer(
                    key="live_ocr",
                    video_processor_factory=LiveOCR,
                    rtc_configuration=RTC,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
            with vc2:
                st.markdown('<div class="sec-title">Live Output</div>', unsafe_allow_html=True)
                if ctx.video_processor:
                    with ctx.video_processor.lock:
                        ltxt = ctx.video_processor.txt
                        lcf = ctx.video_processor.cf
                    st.markdown(metric_card(f"{lcf:.1f}%", "Live Confidence", conf_cls(lcf)), unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f'<div class="result-box">{ltxt or "Point camera at text…"}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="info-box">Start the camera to see live OCR output here.</div>', unsafe_allow_html=True)

        except ImportError:
            st.error("Live stream requires: `pip install streamlit-webrtc av`")
            st.info("Use **Snapshot mode** as an alternative — it works out of the box.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — BATCH PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
with t3:
    st.markdown('<div class="sec-title">Batch Image Processing</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Upload multiple images — OCR runs on all of them. Download a combined report.</div>', unsafe_allow_html=True)

    batch = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="batch_up",
    )

    if batch:
        st.markdown(f"**{len(batch)} file(s) selected**")
        if st.button("🚀 Process All Files", use_container_width=True):
            results = []
            prog = st.progress(0)
            stat_txt = st.empty()

            for i, f in enumerate(batch):
                stat_txt.markdown(f"Processing **{f.name}** ({i+1}/{len(batch)})…")
                img_b = Image.open(f).convert("RGB")
                proc_b = ImagePreprocessor.process(img_b, preprocess_mode)
                data_b, elapsed_b = OCREngine.extract_with_data(proc_b, lang, psm, oem)
                text_b, _ = OCREngine.extract_text(proc_b, lang, psm, oem)
                conf_b = OCREngine.get_confidence(data_b)
                stats_b = OCREngine.text_stats(text_b)
                results.append({
                    "filename": f.name,
                    "text": text_b,
                    "confidence": round(conf_b, 2),
                    "words": stats_b["words"],
                    "characters": stats_b["characters"],
                    "time_ms": round(elapsed_b * 1000, 1),
                })
                st.session_state["history"].append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": f.name,
                    "language": lang_name,
                    "preprocessing": preprocess_mode,
                    "confidence": round(conf_b, 2),
                    "words": stats_b["words"],
                    "characters": stats_b["characters"],
                    "time_ms": round(elapsed_b * 1000, 1),
                })
                prog.progress((i + 1) / len(batch))

            stat_txt.markdown("✅ **All files processed!**")

            df_b = pd.DataFrame([{k: v for k, v in r.items() if k != "text"} for r in results])
            st.markdown('<div class="sec-title">Results Summary</div>', unsafe_allow_html=True)
            st.dataframe(df_b.style.background_gradient(subset=["confidence"], cmap="RdYlGn"), use_container_width=True, hide_index=True)

            avg_cf = df_b["confidence"].mean()
            bs1, bs2, bs3, bs4 = st.columns(4)
            with bs1: st.markdown(metric_card(len(results), "Files"), unsafe_allow_html=True)
            with bs2: st.markdown(metric_card(f"{avg_cf:.1f}%", "Avg Confidence", conf_cls(avg_cf)), unsafe_allow_html=True)
            with bs3: st.markdown(metric_card(df_b["words"].sum(), "Total Words"), unsafe_allow_html=True)
            with bs4: st.markdown(metric_card(f"{df_b['time_ms'].sum():.0f}ms", "Total Time"), unsafe_allow_html=True)

            combined = "\n\n".join(f"=== {r['filename']} ===\n{r['text']}" for r in results)
            bl1, bl2 = st.columns(2)
            with bl1:
                st.download_button("📄 Combined TXT", combined.encode(), "batch_ocr.txt", "text/plain", use_container_width=True)
            with bl2:
                st.download_button("📊 CSV Summary", Exporter.to_csv([{k: v for k, v in r.items() if k != "text"} for r in results]), "batch_summary.csv", "text/csv", use_container_width=True)

            with st.expander("📖 View Individual Results"):
                for r in results:
                    cc = conf_cls(r["confidence"])
                    st.markdown(f'**{r["filename"]}** — <span class="{cc}">{r["confidence"]}% confidence</span> | {r["words"]} words', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-box">{r["text"] or "(no text detected)"}</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — COMPARE PREPROCESSING MODES
# ─────────────────────────────────────────────────────────────────────────────
with t4:
    st.markdown('<div class="sec-title">Preprocessing Mode Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Upload one image to see how all 9 preprocessing modes affect OCR accuracy and visual output side by side.</div>', unsafe_allow_html=True)

    cmp_file = st.file_uploader("Upload for comparison", type=["jpg","jpeg","png","bmp","tiff"], label_visibility="collapsed", key="cmp_up")

    if cmp_file:
        cmp_raw = Image.open(cmp_file).convert("RGB")
        st.image(cmp_raw, caption="Input Image", width=300)

        if st.button("🔬 Run Full Comparison Across All Modes", use_container_width=True):
            cmp_res = {}
            pr = st.progress(0)
            for idx, mode in enumerate(config.PREPROCESS_MODES):
                p = ImagePreprocessor.process(cmp_raw, mode)
                d, elapsed_cmp = OCREngine.extract_with_data(p, lang, psm, oem)
                t, _ = OCREngine.extract_text(p, lang, psm, oem)
                c = OCREngine.get_confidence(d)
                cmp_res[mode] = {"image": p, "text": t, "confidence": c, "words": len(t.split()), "time_ms": elapsed_cmp * 1000}
                pr.progress((idx + 1) / len(config.PREPROCESS_MODES))

            # Summary table
            st.markdown('<div class="sec-title">Accuracy Summary</div>', unsafe_allow_html=True)
            df_cmp = pd.DataFrame([{
                "Mode": m,
                "Confidence (%)": f"{r['confidence']:.1f}",
                "Words Detected": r["words"],
                "Time (ms)": f"{r['time_ms']:.1f}",
            } for m, r in cmp_res.items()])
            st.dataframe(df_cmp, use_container_width=True, hide_index=True)

            best = max(cmp_res, key=lambda m: cmp_res[m]["confidence"])
            st.success(f"🏆 Best mode: **{best}** — {cmp_res[best]['confidence']:.1f}% confidence")

            # Visual grid
            st.markdown('<div class="sec-title">Visual Output Grid</div>', unsafe_allow_html=True)
            modes_list = list(cmp_res.keys())
            for i in range(0, len(modes_list), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(modes_list):
                        m = modes_list[i + j]
                        r = cmp_res[m]
                        icon = "🟢" if r["confidence"] >= 80 else ("🟡" if r["confidence"] >= 60 else "🔴")
                        with col:
                            st.markdown(f"**{m}** {icon} `{r['confidence']:.1f}%`")
                            st.image(r["image"], use_column_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
with t5:
    st.markdown('<div class="sec-title">Session Analytics</div>', unsafe_allow_html=True)

    hist = st.session_state["history"]
    if not hist:
        st.markdown('<div class="info-box">📭 No OCR runs yet this session. Process images to see analytics.</div>', unsafe_allow_html=True)
    else:
        df_h = pd.DataFrame(hist)

        total_r = len(hist)
        avg_cf_h = df_h["confidence"].mean()
        total_w = df_h["words"].sum()
        avg_t = df_h["time_ms"].mean()

        a1, a2, a3, a4 = st.columns(4)
        with a1: st.markdown(metric_card(total_r, "Total Runs"), unsafe_allow_html=True)
        with a2: st.markdown(metric_card(f"{avg_cf_h:.1f}%", "Avg Confidence", conf_cls(avg_cf_h)), unsafe_allow_html=True)
        with a3: st.markdown(metric_card(total_w, "Words Extracted"), unsafe_allow_html=True)
        with a4: st.markdown(metric_card(f"{avg_t:.0f}ms", "Avg Process Time"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown('<div class="sec-title">Confidence Over Runs</div>', unsafe_allow_html=True)
            st.line_chart(df_h[["confidence"]], use_container_width=True)
        with ch2:
            st.markdown('<div class="sec-title">Words per Run</div>', unsafe_allow_html=True)
            st.bar_chart(df_h[["words"]], use_container_width=True)

        if "preprocessing" in df_h.columns:
            pc = df_h.groupby("preprocessing")["confidence"].mean().reset_index()
            st.markdown('<div class="sec-title">Avg Confidence by Preprocessing Mode</div>', unsafe_allow_html=True)
            st.bar_chart(pc.set_index("preprocessing"), use_container_width=True)

        st.markdown('<div class="sec-title">Full Session Log</div>', unsafe_allow_html=True)
        st.dataframe(
            df_h.style.background_gradient(subset=["confidence"], cmap="RdYlGn"),
            use_container_width=True,
            hide_index=True,
        )

        hc1, hc2 = st.columns(2)
        with hc1:
            st.download_button("📊 Export History CSV", Exporter.to_csv(hist), "ocr_session.csv", "text/csv", use_container_width=True)
        with hc2:
            if st.button("🗑️ Clear Session History", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()


with t6:
    st.markdown('<div class="sec-title">🎙️ Speech to Text</div>', unsafe_allow_html=True)

    sl1, sl2 = st.columns([1, 2])

    with sl1:
        st.markdown('<div class="sec-title">Settings</div>', unsafe_allow_html=True)

        stt_lang_name = st.selectbox(
            "🌐 Speech Language",
            list(config.SPEECH_LANGS.keys()),
            key="stt_lang"
        )
        stt_lang = config.SPEECH_LANGS[stt_lang_name]

        lang_flags = {"English": "🇺🇸", "Hindi": "🇮🇳", "German": "🇩🇪", "French": "🇫🇷", "Spanish": "🇪🇸"}
        flag = lang_flags.get(stt_lang_name, "🌐")
        st.markdown(f"""
        <div style="background:#0c0c1a;border:1px solid #1a1a30;border-radius:10px;
        padding:0.8rem;text-align:center;font-size:2rem;margin:0.5rem 0">
            {flag}
            <div style="font-size:0.8rem;color:#a78bfa;font-weight:700;margin-top:0.3rem">{stt_lang_name}</div>
            <div style="font-size:0.7rem;color:#6b7280">{stt_lang}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        stt_mode = st.radio("Input Method", ["🎙️ Live Record", "📁 Upload Audio File"], key="stt_mode")

        st.markdown("""
        <div class="speech-box">
            <b style="color:#f472b6">Tips for best accuracy</b><br>
            · Speak clearly in a quiet room<br>
            · Hold mic 15–30 cm from mouth<br>
            · Speak at normal speed<br>
            · Needs internet (Google API)
        </div>
        """, unsafe_allow_html=True)

    with sl2:
        if stt_mode == "🎙️ Live Record":
            st.warning(
                "🎙️ Live recording is experimental. If recording fails, use Upload Audio File."
            )
            st.markdown('<div class="info-box">Click <b>Start Recording</b>, speak, then click <b>Stop Recording</b>. Results appear automatically.</div>', unsafe_allow_html=True)

            try:
                from streamlit_mic_recorder import mic_recorder

                audio = mic_recorder(
                    start_prompt="⏺  Start Recording",
                    stop_prompt="⏹  Stop Recording",
                    just_once=True,
                    use_container_width=True,
                    key="speech_mic"
                )

                if audio and audio.get("bytes"):
                    if "prev_mic_bytes" not in st.session_state or st.session_state.prev_mic_bytes != audio["bytes"]:
                        st.session_state.prev_mic_bytes = audio["bytes"]
                        with st.spinner(f"Transcribing in {stt_lang_name}…"):
                            stt_text, stt_err = SpeechEngine.transcribe_bytes(audio["bytes"], stt_lang)
                            st.session_state.live_stt_text = stt_text
                            st.session_state.live_stt_err = stt_err
                            if not stt_err:
                                add_history(f"Speech ({stt_lang_name})", stt_lang_name, "Speech Recognition", "Google STT", len(stt_text.split()), len(stt_text), "N/A")

                    if st.session_state.get("live_stt_err"):
                        st.error(f"❌ {st.session_state.live_stt_err}")
                    elif "live_stt_text" in st.session_state:
                        text_output = st.session_state.live_stt_text
                        st.markdown('<div class="sec-title">Transcribed Text</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="result-box">{text_output}</div>', unsafe_allow_html=True)

                        sm1, sm2, sm3 = st.columns(3)
                        with sm1: st.markdown(metric_card(len(text_output.split()), "Words", "c-sp"), unsafe_allow_html=True)
                        with sm2: st.markdown(metric_card(len(text_output), "Characters", "c-sp"), unsafe_allow_html=True)
                        with sm3: st.markdown(metric_card(flag, stt_lang_name), unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        sd1, sd2 = st.columns(2)
                        with sd1:
                            st.download_button("📄 Download TXT", text_output.encode("utf-8"), "speech_transcript.txt", "text/plain", use_container_width=True)
                        with sd2:
                            pdf_s = Exporter.to_pdf_bytes(text_output, {
                                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "confidence": "Google STT",
                                "lang": stt_lang_name,
                                "preprocess": "Speech Recognition",
                                "words": len(text_output.split()),
                            })
                            if pdf_s:
                                st.download_button("📑 Download PDF", pdf_s, "speech_transcript.pdf", "application/pdf", use_container_width=True)

            except ImportError:
                st.error("streamlit-mic-recorder not installed.")
                st.code("pip install streamlit-mic-recorder", language="bash")

        else:
            st.markdown('<div class="info-box">Upload a <b>WAV</b> or <b>FLAC</b> audio file. Record using your phone Voice Memo app and transfer the file.</div>', unsafe_allow_html=True)

            audio_up = st.file_uploader(
                "Upload WAV or FLAC",
                type=["wav", "flac"],
                label_visibility="collapsed",
                key="audio_upload"
            )

            if audio_up:
                st.audio(audio_up, format="audio/wav")
                st.markdown("<br>", unsafe_allow_html=True)

                if st.button("🎯 Transcribe Audio", use_container_width=True):
                    with st.spinner(f"Transcribing in {stt_lang_name}…"):
                        stt_text_u, stt_err_u = SpeechEngine.transcribe_file(
                            audio_up,
                            stt_lang
                        )
                        st.session_state.file_stt_text = stt_text_u
                        st.session_state.file_stt_err = stt_err_u
                        if not stt_err_u:
                            add_history(audio_up.name, stt_lang_name, "Speech Recognition", "Google STT", len(stt_text_u.split()), len(stt_text_u), "N/A")

                if st.session_state.get("file_stt_err"):
                    st.error(f"❌ {st.session_state.file_stt_err}")
                elif "file_stt_text" in st.session_state:
                    file_text = st.session_state.file_stt_text
                    st.markdown('<div class="sec-title">Transcribed Text</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-box">{file_text}</div>', unsafe_allow_html=True)

                    um1, um2 = st.columns(2)
                    with um1: st.markdown(metric_card(len(file_text.split()), "Words", "c-sp"), unsafe_allow_html=True)
                    with um2: st.markdown(metric_card(len(file_text), "Characters", "c-sp"), unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    ud1, ud2 = st.columns(2)
                    with ud1:
                        st.download_button("📄 Download TXT", file_text.encode("utf-8"), "transcript.txt", "text/plain", use_container_width=True)
                    with ud2:
                        pdf_u = Exporter.to_pdf_bytes(file_text, {
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "confidence": "Google STT",
                            "lang": stt_lang_name,
                            "preprocess": "Speech Recognition",
                            "words": len(file_text.split()),
                        })
                        if pdf_u:
                            st.download_button("📑 Download PDF", pdf_u, "transcript.pdf", "application/pdf", use_container_width=True)
