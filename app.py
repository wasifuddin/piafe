import os
import json
import streamlit as st
from PIL import Image

# ── Project imports ──────────────────────────────────────────────────────────
from database.db import init_db
from database.models import (
    save_image, update_image_status,
    save_classification, save_features, save_output,
    get_recent_images, get_stats, get_active_model,
)
from pipeline.validator import validate
from pipeline.preprocessor import preprocess
from pipeline.classifier import classify
from pipeline.feature_extractor import extract
from pipeline.output_generator import generate, to_json, to_csv

# ── One-time DB setup ─────────────────────────────────────────────────────────
init_db()

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PIAFE-AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main .block-container { padding-top: 1.2rem; max-width: 1200px; }

/* ── Login ────────────────────────────────────────────────────── */
.login-logo {
    text-align: center; margin-bottom: 1.5rem; margin-top: 12vh;
}
.login-logo h2 {
    font-size: 1.6rem; font-weight: 800; color: #1B4FD8; margin: 0;
}
.login-logo p {
    font-size: 0.75rem; color: #6B7280; margin: 4px 0 0;
}

/* ── Hero ─────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1B4FD8 0%, #3B82F6 50%, #6366F1 100%);
    border-radius: 14px; padding: 1.8rem 2.2rem; margin-bottom: 1.4rem; color: #fff;
}
.hero h2 { font-size: 1.5rem; font-weight: 800; margin: 0; }
.hero p  { font-size: 0.82rem; opacity: 0.85; margin: 4px 0 0; }

/* ── Cards ────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(145deg, #f8faff, #eef2ff);
    border: 1px solid #dbeafe; border-radius: 14px;
    padding: 1.1rem 1.3rem; text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(27,79,216,0.1); }
.metric-num { font-size: 2rem; font-weight: 800; color: #1B4FD8; }
.metric-lbl { font-size: 0.65rem; color: #6B7280; font-weight: 600; text-transform: uppercase; letter-spacing: .07em; margin-top: 3px; }

.meta-card {
    background: #f8faff; border: 1px solid #e0e7ff;
    border-radius: 12px; padding: 1rem 1.2rem;
}
.meta-row { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #f0f4ff; font-size: 0.84rem; }
.meta-row:last-child { border-bottom: none; }
.meta-key { color: #6B7280; font-weight: 500; }
.meta-val { color: #1e293b; font-weight: 600; }

.room-badge {
    display: inline-block; background: linear-gradient(135deg, #1B4FD8, #3B82F6);
    color: #fff; padding: 6px 20px; border-radius: 24px;
    font-size: 0.85rem; font-weight: 700; letter-spacing: .04em;
    margin-bottom: 8px; box-shadow: 0 3px 10px rgba(27,79,216,0.2);
}
.conf-big { font-size: 2.6rem; font-weight: 800; color: #1B4FD8; line-height: 1; margin: 4px 0; }
.conf-lbl { color: #6B7280; font-size: 0.78rem; }

.feat-card {
    background: linear-gradient(145deg, #f8faff, #f0f4ff);
    border: 1px solid #e0e7ff; border-radius: 14px;
    padding: 0.95rem 1.15rem; margin-bottom: 10px;
    transition: transform 0.2s, box-shadow 0.2s;
}
.feat-card:hover { transform: translateY(-1px); box-shadow: 0 5px 18px rgba(27,79,216,0.08); }
.feat-title { font-size: 0.63rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; color: #6B7280; margin-bottom: 2px; }
.feat-val   { font-size: 1.1rem; font-weight: 700; color: #111827; }
.feat-conf.good { font-size: 0.75rem; font-weight: 600; color: #059669; margin-top: 2px; }
.feat-conf.warn { font-size: 0.75rem; font-weight: 600; color: #D97706; margin-top: 2px; }
.warn-badge { display: inline-block; background: #FEF3C7; color: #92400E; font-size: 0.63rem; font-weight: 700; padding: 2px 10px; border-radius: 10px; margin-top: 3px; }

.section-hdr { font-size: 1.05rem; font-weight: 700; color: #1e293b; margin-bottom: 0.5rem; padding-bottom: 0.4rem; border-bottom: 2px solid #e0e7ff; }
.step-done { color: #059669; font-weight: 600; }
.step-run  { color: #1B4FD8; font-weight: 600; }
.step-err  { color: #DC2626; font-weight: 600; }

.score-label { font-size: 0.78rem; min-width: 110px; color: #6B7280; font-weight: 500; }
.score-label.active { color: #1B4FD8; font-weight: 700; }

/* ── Sidebar ──────────────────────────────────────────────────── */
.sidebar-brand {
    background: linear-gradient(135deg, #1B4FD8, #6366F1);
    padding: 1.1rem 1rem; margin: -1rem -1rem 1rem; border-radius: 0 0 12px 12px;
    color: #fff; text-align: center;
}
.sidebar-brand h3 { margin: 0; font-size: 1.15rem; font-weight: 800; }
.sidebar-brand p  { margin: 2px 0 0; font-size: 0.68rem; opacity: 0.8; }
.model-card {
    background: #f8faff; border: 1px solid #e0e7ff;
    border-radius: 10px; padding: 0.75rem 0.9rem; margin-top: 0.8rem;
}
.mi-label { font-size: 0.62rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; color: #6B7280; }
.mi-value { font-size: 0.82rem; font-weight: 600; color: #1e293b; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════ #
#  LOGIN GATE
# ═══════════════════════════════════════════════════════════════════════════ #

VALID_USER = "project"
VALID_PASS = "altair123"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    # Hide sidebar on login page
    st.markdown("<style>[data-testid='stSidebar']{display:none}</style>", unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        st.markdown("""
        <div class="login-logo">
            <h2>PIAFE-AI</h2>
            <p>Property Image Analysis and Feature Extraction</p>
            <p style="margin-top:2px">ICT802 Capstone · Tech Adaptive · VIT</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

            if submitted:
                if username == VALID_USER and password == VALID_PASS:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════ #
#  SIDEBAR (shown only after login)
# ═══════════════════════════════════════════════════════════════════════════ #

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h3>PIAFE-AI</h3>
        <p>ICT802 Capstone · Tech Adaptive · VIT</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", ["Analyze Image", "Dashboard"], label_visibility="collapsed")
    st.divider()

    model = get_active_model()
    st.markdown(f"""
    <div class="model-card">
        <div class="mi-label">Active Model</div>
        <div class="mi-value">{model.get('model_name','ResNet-50')}</div>
        <div style="margin-top:5px"><div class="mi-label">Version</div><div class="mi-value">{model.get('version_number','v1.0.0')}</div></div>
        <div style="margin-top:5px"><div class="mi-label">Framework</div><div class="mi-value">{model.get('framework','TensorFlow')}</div></div>
        <div style="margin-top:5px"><div class="mi-label">Accuracy</div><div class="mi-value">{f"{model.get('accuracy_score',0)*100:.1f}%" if model.get('accuracy_score') else 'Prototype mode'}</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("Sign Out", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════ #
#  PAGE 1 — Analyze Image
# ═══════════════════════════════════════════════════════════════════════════ #

if "Analyze Image" in page:

    st.markdown("""
    <div class="hero">
        <h2>Property Image Analysis</h2>
        <p>Upload a property photo and the AI pipeline will classify the room and extract visual features.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a property image",
        type=["jpg", "jpeg", "png", "webp", "tiff"],
        help="Minimum 50x50 pixels. JPG, PNG, WEBP, TIFF accepted.",
    )

    if not uploaded_file:
        st.info("Upload a property image above to begin analysis.")
        st.stop()

    ext = os.path.splitext(uploaded_file.name)[1]
    tmp_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())

    col_img, col_meta = st.columns([1, 1])
    with col_img:
        st.image(tmp_path, caption=uploaded_file.name, use_column_width=True)

    st.divider()
    run = st.button("Run Analysis Pipeline", type="primary", use_container_width=True)

    if not run:
        st.stop()

    # ── Pipeline ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Pipeline Progress</div>', unsafe_allow_html=True)
    prog = st.progress(0)
    log  = st.empty()

    def log_msg(msg, kind="run"):
        css = {"run": "step-run", "done": "step-done", "err": "step-err"}.get(kind, "step-run")
        icon = {"run": "▸", "done": "✓", "err": "✗"}.get(kind, "▸")
        log.markdown(f'<span class="{css}">{icon} {msg}</span>', unsafe_allow_html=True)

    # Stage 1
    log_msg("Stage 1: Validating image…")
    ok, msg, meta = validate(tmp_path)
    if not ok:
        log_msg(f"Validation failed: {msg}", "err")
        st.error(msg)
        st.stop()
    prog.progress(20)
    log_msg(f"Stage 1 complete — {meta['width']}x{meta['height']}px {meta['format']}", "done")

    with col_meta:
        st.markdown(f"""
        <div class="meta-card">
            <div style="font-weight:700; font-size:0.88rem; margin-bottom:8px;">Image Metadata</div>
            <div class="meta-row"><span class="meta-key">Format</span><span class="meta-val">{meta['format']}</span></div>
            <div class="meta-row"><span class="meta-key">Resolution</span><span class="meta-val">{meta['width']} x {meta['height']} px</span></div>
            <div class="meta-row"><span class="meta-key">File Size</span><span class="meta-val">{meta['file_size_kb']} KB</span></div>
            <div class="meta-row"><span class="meta-key">Colour Mode</span><span class="meta-val">{meta['mode']}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # Stage 2
    log_msg("Stage 2: Pre-processing image…")
    image_id = save_image(tmp_path, meta["format"], meta["width"], meta["height"])
    update_image_status(image_id, "processing")
    array, pil_224 = preprocess(tmp_path)
    prog.progress(40)
    log_msg("Stage 2 complete — resized, normalised, brightness corrected.", "done")

    # Stage 3
    log_msg("Stage 3: Running room classification (ResNet-50)…")
    with st.spinner("Loading model and running inference…"):
        classification = classify(array)
    prog.progress(65)
    log_msg(f"Stage 3 complete — {classification['room_type']} ({classification['confidence']*100:.1f}% confidence)", "done")
    save_classification(image_id, classification["room_type"], classification["confidence"], classification["all_scores"])

    # Stage 4
    log_msg("Stage 4: Extracting visual features…")
    features = extract(pil_224)
    prog.progress(85)
    log_msg("Stage 4 complete — flooring, lighting, windows, condition detected.", "done")
    save_features(image_id, features)

    # Stage 5
    log_msg("Stage 5: Generating structured output files…")
    result = generate(image_id, uploaded_file.name, classification, features)
    json_path = to_json(result, image_id)
    csv_path  = to_csv(result, image_id)
    save_output(image_id, "JSON", json_path)
    save_output(image_id, "CSV",  csv_path)
    update_image_status(image_id, "complete")
    prog.progress(100)
    log_msg("Stage 5 complete — JSON + CSV files ready for download.", "done")

    # ── Results ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-hdr">Analysis Results</div>', unsafe_allow_html=True)

    room = classification["room_type"]
    conf = classification["confidence"]
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### Room Classification")
        st.markdown(
            f'<div class="room-badge">{room}</div>'
            f'<div class="conf-big">{conf*100:.1f}%</div>'
            f'<div class="conf-lbl">Classification confidence</div>',
            unsafe_allow_html=True,
        )
        if conf < 0.60:
            st.warning("Confidence below 60% — manual review recommended.")

        st.markdown("**All room scores**")
        for label, score in sorted(classification["all_scores"].items(), key=lambda x: -x[1]):
            is_top = label == room
            cls = "score-label active" if is_top else "score-label"
            st.markdown(f'<span class="{cls}">{label}</span>', unsafe_allow_html=True)
            st.progress(score, text=f"{int(score*100)}%")

    with c2:
        st.markdown("#### Visual Features")
        feat_items = [
            ("Flooring Type",      features["flooring"],  ),
            ("Lighting Quality",   features["lighting"],  ),
            ("Window Presence",    features["window"],    ),
            ("Property Condition", features["condition"], ),
        ]
        for label, feat in feat_items:
            conf_f = feat["confidence"]
            warn = conf_f < 0.70
            ccls = "feat-conf warn" if warn else "feat-conf good"
            whtml = '<div class="warn-badge">Review recommended</div>' if warn else ""
            st.markdown(
                f'<div class="feat-card">'
                f'<div class="feat-title">{label}</div>'
                f'<div class="feat-val">{feat["value"]}</div>'
                f'<div class="{ccls}">{conf_f*100:.1f}% confidence</div>'
                f'{whtml}</div>',
                unsafe_allow_html=True,
            )

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-hdr">Export Structured Output</div>', unsafe_allow_html=True)

    tab_json, tab_csv = st.tabs(["JSON", "CSV"])
    with tab_json:
        st.code(json.dumps(result, indent=2), language="json")
        with open(json_path, "rb") as f:
            st.download_button("Download JSON", f, file_name=f"piafe_{image_id}.json", mime="application/json")
    with tab_csv:
        import pandas as pd
        flat = {
            "image_id": result["image_id"], "file_name": result["file_name"],
            "analysed_at": result["analysed_at"], "room_type": result["room_type"],
            "room_confidence": result["room_confidence"],
            **{f"score_{k.lower().replace(' ','_')}": v for k, v in result["all_room_scores"].items()},
            "flooring_type": result["features"]["flooring_type"],
            "flooring_conf": result["features"]["flooring_conf"],
            "lighting_quality": result["features"]["lighting_quality"],
            "lighting_conf": result["features"]["lighting_conf"],
            "window_present": result["features"]["window_present"],
            "window_conf": result["features"]["window_conf"],
            "property_condition": result["features"]["property_condition"],
            "condition_conf": result["features"]["condition_conf"],
        }
        st.dataframe(pd.DataFrame([flat]), use_container_width=True)
        with open(csv_path, "rb") as f:
            st.download_button("Download CSV", f, file_name=f"piafe_{image_id}.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════ #
#  PAGE 2 — Dashboard
# ═══════════════════════════════════════════════════════════════════════════ #

elif "Dashboard" in page:
    import pandas as pd

    st.markdown("""
    <div class="hero">
        <h2>System Dashboard</h2>
        <p>Overview of all analyses, model performance, and historical results.</p>
    </div>
    """, unsafe_allow_html=True)

    stats = get_stats()
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-num">{stats["total_images"]}</div><div class="metric-lbl">Images Processed</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-num">{stats["avg_confidence"]}%</div><div class="metric-lbl">Avg Confidence</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-num">{stats["total_outputs"]}</div><div class="metric-lbl">Output Files</div></div>', unsafe_allow_html=True)
    with m4:
        model = get_active_model()
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="font-size:1.3rem">{model.get("version_number","v1.0.0")}</div><div class="metric-lbl">Active Model</div></div>', unsafe_allow_html=True)

    st.divider()
    col_chart, col_model = st.columns([1, 1])

    with col_chart:
        st.markdown('<div class="section-hdr">Room Type Distribution</div>', unsafe_allow_html=True)
        dist = stats.get("room_distribution", {})
        if dist:
            df_dist = pd.DataFrame(list(dist.items()), columns=["Room Type", "Count"]).sort_values("Count", ascending=False)
            st.bar_chart(df_dist.set_index("Room Type"))
        else:
            st.info("No analyses yet. Upload an image to get started.")

    with col_model:
        st.markdown('<div class="section-hdr">Active Model</div>', unsafe_allow_html=True)
        m = get_active_model()
        st.markdown(f"""
        <div class="meta-card">
            <div class="meta-row"><span class="meta-key">Name</span><span class="meta-val">{m.get('model_name','ResNet-50 Room Classifier')}</span></div>
            <div class="meta-row"><span class="meta-key">Version</span><span class="meta-val">{m.get('version_number','v1.0.0')}</span></div>
            <div class="meta-row"><span class="meta-key">Framework</span><span class="meta-val">{m.get('framework','TensorFlow')}</span></div>
            <div class="meta-row"><span class="meta-key">Accuracy</span><span class="meta-val">{f"{m.get('accuracy_score',0)*100:.1f}%" if m.get('accuracy_score') else '— (prototype)'}</span></div>
            <div class="meta-row"><span class="meta-key">Deployed</span><span class="meta-val">{m.get('deployed_date','—')}</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-hdr">Recent Analyses</div>', unsafe_allow_html=True)
    rows = get_recent_images(20)
    if rows:
        df = pd.DataFrame(rows)
        keep = ["image_id", "file_path", "file_format", "width", "height",
                "status", "room_type", "confidence_score",
                "flooring_type", "lighting_quality", "window_present", "property_condition"]
        df = df[[c for c in keep if c in df.columns]]
        df["file_path"] = df["file_path"].apply(os.path.basename)
        if "confidence_score" in df.columns:
            df["confidence_score"] = df["confidence_score"].apply(lambda x: f"{x*100:.1f}%" if x else "—")
        df.columns = [c.replace("_", " ").title() for c in df.columns]
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No images have been processed yet.")
