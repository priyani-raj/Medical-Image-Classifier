"""
app.py — Streamlit Frontend
Medical Image Classification System
JIIT Noida | AI & ML Lab | 3rd Semester

"""

import os
import sys
import tempfile
import numpy as np
import streamlit as st
from PIL import Image

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "MedScan AI",
    page_icon  = "🩺",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ─── STYLES ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root & background ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0e17;
    color: #e8eaf0;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #0f1520;
    border-right: 1px solid #1e2d45;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Typography ── */
h1, h2, h3 { font-family: 'Space Mono', monospace; letter-spacing: -0.5px; }

/* ── Hero title ── */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: #5a7a9a;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* ── Model cards ── */
.model-card {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
    cursor: pointer;
}
.model-card:hover { border-color: #2563eb; }
.model-card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    color: #3b82f6;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.model-card-desc {
    font-size: 0.82rem;
    color: #6b7a90;
    margin-top: 0.2rem;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #1e3a5f !important;
    border-radius: 12px !important;
    background: #0d1623 !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #2563eb !important;
}

/* ── Result box ── */
.result-box {
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-top: 1.2rem;
    border: 1px solid;
}
.result-normal {
    background: linear-gradient(135deg, #0a2a1a, #0d1f12);
    border-color: #16a34a;
}
.result-abnormal {
    background: linear-gradient(135deg, #2a0a0a, #1f0d0d);
    border-color: #dc2626;
}
.result-mild {
    background: linear-gradient(135deg, #1a1a0a, #1f1f0d);
    border-color: #ca8a04;
}
.result-severe {
    background: linear-gradient(135deg, #2a0a1a, #1f0d16);
    border-color: #9333ea;
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-model {
    font-size: 0.75rem;
    color: #5a7a9a;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
}

/* ── Confidence bar ── */
.conf-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #5a7a9a;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    margin-top: 1rem;
}

/* ── Grade pills (eye model) ── */
.grade-pill {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    margin-right: 0.3rem;
    margin-top: 0.3rem;
}

/* ── Sidebar selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #111827 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #1d4ed8;
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 0.6rem 1.4rem;
    transition: background 0.2s;
    width: 100%;
}
.stButton > button:hover { background: #2563eb; }

/* ── Status badge ── */
.status-badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.status-ready   { background: #14532d; color: #4ade80; }
.status-pending { background: #422006; color: #fb923c; }

/* ── Divider ── */
hr { border-color: #1e2d45 !important; }

/* ── Info box ── */
.info-note {
    background: #0d1a2e;
    border-left: 3px solid #2563eb;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    font-size: 0.82rem;
    color: #7a9cc0;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─── MODEL STATUS CHECK ───────────────────────────────────────────────────────
MODEL_FILES = {
    "chest"    : "results/chest_model.h5",
    "fracture" : "results/fracture_model.h5",
    "eye"      : "results/eye_model.h5",
    "brain"    : "results/brain_model.h5",
}

def model_ready(key):
    return os.path.exists(MODEL_FILES[key])

def status_badge(key):
    if model_ready(key):
        return '<span class="status-badge status-ready">● READY</span>'
    return '<span class="status-badge status-pending">○ NOT TRAINED</span>'


# ─── PREDICTION ───────────────────────────────────────────────────────────────
def run_prediction(img_path, model_type):
    try:
        sys.path.insert(0, os.getcwd())
        from router import predict_image
        return predict_image(img_path, model_type)
    except Exception as e:
        return {"error": str(e), "label": "Error", "confidence": 0}


# ─── RESULT COLOUR LOGIC ─────────────────────────────────────────────────────
def result_css_class(model_type, label, grade=None):
    if model_type == "eye":
        if grade == 0:   return "result-normal"
        if grade in (1, 2): return "result-mild"
        return "result-severe"
    label_up = label.upper()
    if "NORMAL" in label_up or "NO " in label_up:
        return "result-normal"
    return "result-abnormal"

def grade_color(grade):
    colors = {0:"#16a34a", 1:"#ca8a04", 2:"#d97706", 3:"#dc2626", 4:"#9333ea"}
    return colors.get(grade, "#6b7a90")


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:1.5rem'>
        <div style='font-family:Space Mono,monospace;font-size:1.1rem;
                    font-weight:700;color:#fff;letter-spacing:-0.5px'>
            🩺 MedScan AI
        </div>
        <div style='font-size:0.72rem;color:#3b6a9a;letter-spacing:2px;
                    text-transform:uppercase;margin-top:0.1rem'>
            JIIT Noida · AI & ML Lab
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("**SELECT MODEL**", )

    MODEL_INFO = {
        "🫁  Chest X-Ray"     : ("chest",    "Pneumonia Detection",          "VGG16"),
        "🦴  Bone Fracture"   : ("fracture", "Fracture Detection",            "ResNet50"),
        "👁️  Retinal Scan"    : ("eye",      "Diabetic Retinopathy Grading",  "EfficientNetB3"),
        "🧠  Brain MRI"       : ("brain",    "Tumor Classification",          "EfficientNetB0"),
    }

    selected_display = st.selectbox(
        "Model", list(MODEL_INFO.keys()), label_visibility="collapsed"
    )
    model_type, model_desc, backbone = MODEL_INFO[selected_display]

    st.markdown(f"""
    <div class="model-card">
        <div class="model-card-title">{backbone}</div>
        <div class="model-card-desc">{model_desc}</div>
        <div style='margin-top:0.5rem'>{status_badge(model_type)}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Model status overview
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:0.7rem;
                color:#3b6a9a;letter-spacing:2px;text-transform:uppercase;
                margin-bottom:0.8rem'>ALL MODELS</div>
    """, unsafe_allow_html=True)

    for display, (key, desc, bb) in MODEL_INFO.items():
        icon = display.split()[0]
        name = " ".join(display.split()[1:])
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;
                    align-items:center;margin-bottom:0.4rem;
                    padding:0.4rem 0.6rem;border-radius:8px;
                    background:#111827;border:1px solid #1a2535'>
            <span style='font-size:0.8rem;color:#c0cfe0'>{icon} {name}</span>
            {status_badge(key)}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem;color:#3b5a7a;line-height:1.6'>
        Team: Ishita · Priyani<br>Hrishita · Samman<br>
        <span style='color:#1e3a5f'>3rd Semester · 2025</span>
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown(f"""
    <div class="hero-title">Medical<br>Image Scanner</div>
    <div class="hero-sub">Diagnostic assistance</div>
    """, unsafe_allow_html=True)

    # Upload
    st.markdown(f"""
    <div style='font-family:Space Mono,monospace;font-size:0.72rem;
                color:#3b6a9a;letter-spacing:2px;text-transform:uppercase;
                margin-bottom:0.5rem'>
        UPLOAD — {selected_display.strip()} 
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop image here",
        type        = ["jpg", "jpeg", "png"],
        label_visibility = "collapsed"
    )

    if not model_ready(model_type):
        st.markdown(f"""
        <div class="info-note">
            ⚠ <strong>{backbone}</strong> model not trained yet.<br>
            Run: <code>python train.py --model {model_type}</code>
        </div>
        """, unsafe_allow_html=True)

    if uploaded:
        st.markdown("<div style='margin-top:1rem'>", unsafe_allow_html=True)
        btn = st.button("⚡  Run Analysis", disabled=not model_ready(model_type))
        st.markdown("</div>", unsafe_allow_html=True)

        if btn:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            with st.spinner("Analysing image..."):
                result = run_prediction(tmp_path, model_type)
            os.unlink(tmp_path)

            if "error" in result and result["label"] == "Error":
                st.error(f"Prediction failed: {result['error']}")
            else:
                grade    = result.get("grade", None)
                css_cls  = result_css_class(model_type, result["label"], grade)
                conf     = result.get("confidence", 0)

                st.markdown(f"""
                <div class="result-box {css_cls}">
                    <div class="result-model">{result.get('model_used','—')}</div>
                    <div class="result-label">{result['label']}</div>
                    {"<div style='font-size:0.82rem;color:#9aa8b8;margin-top:0.3rem'>" + result.get('note','') + "</div>" if result.get('note') else ''}
                </div>
                """, unsafe_allow_html=True)

                # Confidence bar
                st.markdown('<div class="conf-label">Confidence</div>', unsafe_allow_html=True)
                st.progress(int(conf) / 100)
                st.markdown(f"""
                <div style='font-family:Space Mono,monospace;font-size:1rem;
                            font-weight:700;color:#e8eaf0;margin-top:0.2rem'>
                    {conf:.1f}%
                </div>
                """, unsafe_allow_html=True)

                # Eye model: show per-grade probability bars + clinical advice
                if model_type == "eye" and grade is not None:
                    # ── Clinical advice ──────────────────────────────────
                    advice = result.get("advice", "")
                    if advice:
                        st.markdown(f"""
                        <div class="info-note" style="margin-top:1rem">
                            💡 <strong>Clinical note:</strong> {advice}
                        </div>
                        """, unsafe_allow_html=True)

                    # ── DR Grade probability breakdown ───────────────────
                    st.markdown(
                        '<div class="conf-label" style="margin-top:1.2rem">'
                        'DR Grade Probability Breakdown</div>',
                        unsafe_allow_html=True,
                    )
                    grade_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
                    all_probs    = result.get("all_probs", [])

                    if all_probs:
                        # Horizontal bar for each grade
                        bars_html = ""
                        for i, (g, prob) in enumerate(zip(grade_labels, all_probs)):
                            col  = grade_color(i)
                            bold = "font-weight:700;" if i == grade else "opacity:0.75;"
                            bars_html += f"""
                            <div style="margin-bottom:0.45rem">
                              <div style="display:flex;justify-content:space-between;
                                          font-family:'Space Mono',monospace;
                                          font-size:0.72rem;margin-bottom:0.15rem;{bold}
                                          color:{col}">
                                <span>{'▶ ' if i==grade else ''}{g}</span>
                                <span>{prob:.1f}%</span>
                              </div>
                              <div style="background:#1a2535;border-radius:4px;height:7px;overflow:hidden">
                                <div style="width:{prob}%;height:100%;
                                            background:{col};
                                            border-radius:4px;
                                            transition:width 0.4s ease">
                                </div>
                              </div>
                            </div>"""
                        st.markdown(bars_html, unsafe_allow_html=True)
                    else:
                        # Fallback: grade pills (no probabilities returned)
                        pills = ""
                        for i, g in enumerate(grade_labels):
                            pills += f"""<span class="grade-pill"
                                style="background:{grade_color(i)}22;
                                       color:{grade_color(i)};
                                       border:1px solid {grade_color(i)}55;
                                       {('font-size:0.82rem;border-width:2px' if i==grade else '')}">
                                {'▶ ' if i==grade else ''}{g}
                            </span>"""
                        st.markdown(pills, unsafe_allow_html=True)


with col_right:
    if uploaded:
        img = Image.open(uploaded)
        st.markdown("""
        <div style='font-family:Space Mono,monospace;font-size:0.7rem;
                    color:#3b6a9a;letter-spacing:2px;text-transform:uppercase;
                    margin-bottom:0.5rem'>UPLOADED IMAGE</div>
        """, unsafe_allow_html=True)
        st.image(img, use_container_width=True)

        # Image metadata
        w, h = img.size
        mode = img.mode
        st.markdown(f"""
        <div style='display:flex;gap:1rem;margin-top:0.5rem'>
            <div style='background:#111827;border:1px solid #1e2d45;border-radius:8px;
                        padding:0.5rem 0.9rem;flex:1;text-align:center'>
                <div style='font-family:Space Mono,monospace;font-size:0.65rem;
                            color:#3b6a9a;letter-spacing:1px'>WIDTH</div>
                <div style='font-family:Space Mono,monospace;font-size:0.95rem;
                            font-weight:700;color:#e8eaf0'>{w}px</div>
            </div>
            <div style='background:#111827;border:1px solid #1e2d45;border-radius:8px;
                        padding:0.5rem 0.9rem;flex:1;text-align:center'>
                <div style='font-family:Space Mono,monospace;font-size:0.65rem;
                            color:#3b6a9a;letter-spacing:1px'>HEIGHT</div>
                <div style='font-family:Space Mono,monospace;font-size:0.95rem;
                            font-weight:700;color:#e8eaf0'>{h}px</div>
            </div>
            <div style='background:#111827;border:1px solid #1e2d45;border-radius:8px;
                        padding:0.5rem 0.9rem;flex:1;text-align:center'>
                <div style='font-family:Space Mono,monospace;font-size:0.65rem;
                            color:#3b6a9a;letter-spacing:1px'>MODE</div>
                <div style='font-family:Space Mono,monospace;font-size:0.95rem;
                            font-weight:700;color:#e8eaf0'>{mode}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Empty state
        st.markdown("""
        <div style='height:380px;border:1px solid #1e2d45;border-radius:14px;
                    display:flex;flex-direction:column;align-items:center;
                    justify-content:center;background:#0d1623;text-align:center'>
            <div style='font-size:3rem;margin-bottom:0.8rem;opacity:0.3'>🩻</div>
            <div style='font-family:Space Mono,monospace;font-size:0.75rem;
                        color:#2a4a6a;letter-spacing:2px;text-transform:uppercase'>
                No image uploaded
            </div>
            <div style='font-size:0.8rem;color:#1e3a5a;margin-top:0.4rem'>
                Upload an image to begin analysis
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── BOTTOM INFO BAR ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='display:flex;justify-content:space-between;align-items:center;
            font-size:0.72rem;color:#2a4a6a;font-family:Space Mono,monospace;
            letter-spacing:0.5px;padding-bottom:0.5rem'>
    <span>MedScan AI · JIIT Noida · AI &amp; ML Lab · 3rd Semester</span>
    <span style='color:#1e3050'>⚠ For educational use only — not a medical device</span>
</div>
""", unsafe_allow_html=True)