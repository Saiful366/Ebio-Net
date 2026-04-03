import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

# ── Set theme inline — no config.toml needed ──
st.config.set_option("theme.base", "dark")
st.config.set_option("theme.primaryColor", "#ef4444")
st.config.set_option("theme.backgroundColor", "#020617")
st.config.set_option("theme.secondaryBackgroundColor", "#1e293b")
st.config.set_option("theme.textColor", "#f8fafc")

st.set_page_config(
    page_title="Ebio-Net | Professional Edition",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

    .stApp {
        background-color: #020617;
        background-image:
            radial-gradient(at 0% 0%, rgba(30,41,59,0.5) 0, transparent 50%),
            radial-gradient(at 50% 0%, rgba(15,23,42,0.3) 0, transparent 50%),
            radial-gradient(at 100% 0%, rgba(239,68,68,0.05) 0, transparent 50%);
    }

    .nav-header {
        position: fixed; top:0; left:0; right:0; height:70px;
        background: rgba(15,23,42,0.9); backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(255,255,255,0.05);
        display:flex; align-items:center; justify-content:space-between;
        padding: 0 4rem; z-index:1000;
    }
    .nav-logo { font-weight:800; font-size:1.5rem; color:#fff; }
    .nav-logo span { color:#ef4444; }
    .nav-status {
        font-size:0.8rem; color:#94a3b8; background:rgba(255,255,255,0.05);
        padding:4px 12px; border-radius:20px; border:1px solid rgba(255,255,255,0.1);
    }

    .hero-container { padding:120px 0 40px 0; text-align:center; }
    .hero-title { font-size:3.5rem; font-weight:800; letter-spacing:-2px; margin:0; color:#fff; }
    .step-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:20px; max-width:1000px; margin:40px auto; }
    .step-card { background:rgba(30,41,59,0.5); border:1px solid rgba(255,255,255,0.05); padding:1.5rem; border-radius:20px; }
    .step-num { font-size:0.8rem; font-weight:800; color:#ef4444; display:block; margin-bottom:8px; }

    .stTabs [data-baseweb="tab-list"] { background:rgba(255,255,255,0.03); padding:8px; border-radius:16px; margin-bottom:30px; }
    .stTabs [aria-selected="true"] { background:#ffffff !important; color:#020617 !important; }

    /* Dropzone */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(30,41,59,0.6) !important;
        border: 2px dashed rgba(239,68,68,0.5) !important;
        border-radius: 16px !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #ef4444 !important;
        background: rgba(239,68,68,0.05) !important;
    }

    /* Browse files → GREEN */
    [data-testid="stFileUploadDropzone"] button,
    [data-testid="stFileUploadDropzone"] button *  {
        background: #16a34a !important;
        background-color: #16a34a !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }

    /* Download button → BLUE */
    [data-testid="stDownloadButton"] button,
    [data-testid="stDownloadButton"] button * {
        background: #2563eb !important;
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }
    [data-testid="stDownloadButton"] button:hover,
    [data-testid="stDownloadButton"] button:hover * {
        background: #1d4ed8 !important;
        background-color: #1d4ed8 !important;
    }

    /* Clear buttons → red outline, via ID wrapper */
    #clear_single_wrap button, #clear_single_wrap button *,
    #clear_batch_wrap button,  #clear_batch_wrap button * {
        background: transparent !important;
        background-color: transparent !important;
        color: #ef4444 !important;
        border: 1.5px solid #ef4444 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }

    /* Confirm directory → TEAL, via ID wrapper */
    #confirm_dir_wrap button, #confirm_dir_wrap button * {
        background: #0d9488 !important;
        background-color: #0d9488 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }


    .stTextInput > div > div > input {
        background: rgba(30,41,59,0.8) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #f8fafc !important; border-radius: 12px !important;
    }
    .footer { text-align:center; padding:4rem; color:#475569; border-top:1px solid rgba(255,255,255,0.05); }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="nav-header">
        <div class="nav-logo">🌿 EBIO<span>-NET</span></div>
        <div class="nav-status">● SYSTEM READY | INDUSTRIAL ENGINE</div>
    </div>
    <div class="hero-container">
        <h1 class="hero-title">🌿 Ebio-Net</h1>
        <div class="step-grid">
            <div class="step-card"><span class="step-num">01</span><p>Upload a sample image or link a local data folder.</p></div>
            <div class="step-card"><span class="step-num">02</span><p>The neural engine performs a full spectral classification scan.</p></div>
            <div class="step-card"><span class="step-num">03</span><p>Review the visual reports and download the data telemetry.</p></div>
        </div>
    </div>
""", unsafe_allow_html=True)

CLASSES = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")

@st.cache_resource
def load_model():
    path = "restored_best_model_88.keras"
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), path)
    return tf.keras.models.load_model(path)

def process_img(image, model):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    arr = np.expand_dims(np.array(img).astype(np.float32), axis=0)
    return model.predict(preprocess_input(arr), verbose=0)[0]

def create_labeled_image(original_img, label, confidence):
    img = original_img.copy()
    draw = ImageDraw.Draw(img)
    font_size = max(14, int(img.size[0] * 0.05))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    text = f"{label} ({confidence:.1%})"
    bbox = draw.textbbox((20, 20), text, font=font)
    draw.rectangle([bbox[0]-10, bbox[1]-5, bbox[2]+10, bbox[3]+5], fill="#ef4444")
    draw.text((20, 20), text, font=font, fill="white")
    return img

def list_images_in_dir(dir_path):
    try:
        return sorted([f for f in os.listdir(dir_path) if f.lower().endswith(IMG_EXTS)])
    except:
        return []

for k, v in [('single_key', 0), ('batch_key', 100), ('b_path', '')]:
    if k not in st.session_state:
        st.session_state[k] = v

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
    model = None

if model:
    tabs = st.tabs(["🎯 PRECISION SCAN", "🚀 BATCH PIPELINE"])

    # ══ TAB 1 — PRECISION SCAN ══
    with tabs[0]:
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown("### Sample Input")

            st.markdown('<div id="clear_single_wrap">', unsafe_allow_html=True)
            if st.button("✕ Clear image", key="clear_single"):
                st.session_state['single_key'] += 1
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            file = st.file_uploader(
                "Select Image",
                type=["jpg", "png", "jpeg", "webp", "tiff", "tif"],
                label_visibility="collapsed",
                key=f"sup_{st.session_state['single_key']}"
            )

            if file:
                raw_img = Image.open(file)
                st.image(raw_img, use_container_width=True, caption="Sample Ready")
                run = st.button("⚡ EXECUTE NEURAL SCAN", type="primary", use_container_width=True)
            else:
                run = False

        with c2:
            if file and run:
                with st.spinner("Analyzing spectral data..."):
                    probs = process_img(raw_img, model)
                    idx = int(np.argmax(probs))
                    label = CLASSES[idx]
                    conf = float(probs[idx])
                    st.markdown("### Intelligence Report")
                    labeled = create_labeled_image(raw_img, label, conf)
                    st.image(labeled, use_container_width=True, caption="Classified Asset")
                    st.write("---")
                    st.write(f"**Primary Match:** {label} ({conf:.2%})")
                    for i, c in enumerate(CLASSES):
                        st.write(f"{c}")
                        st.progress(float(probs[i]))
            else:
                st.info("Input a leaf sample to reveal classification data.")

    # ══ TAB 2 — BATCH PIPELINE ══
    with tabs[1]:
        st.markdown("### Mass Classification Pipeline")
        b1, b2 = st.columns(2)

        with b1:
            st.markdown("**📁 Link Local Directory**")
            st.caption("Paste the full folder path below.")

            typed_path = st.text_input(
                "Folder path",
                value=st.session_state.get("b_path", ""),
                placeholder="/Users/saiful/Downloads/images",
                label_visibility="collapsed"
            )

            st.markdown('<div id="confirm_dir_wrap">', unsafe_allow_html=True)
            if st.button("✅ CONFIRM DIRECTORY", use_container_width=True, key="confirm_dir"):
                if typed_path and os.path.isdir(typed_path):
                    st.session_state["b_path"] = typed_path
                    st.success(f"📍 Linked: {os.path.basename(typed_path)}")
                elif typed_path:
                    st.error("⚠️ Path not found. Check and try again.")
            st.markdown('</div>', unsafe_allow_html=True)

            active = st.session_state.get("b_path", "")
            if active and os.path.isdir(active):
                st.success(f"📍 Active: {active}")
                files = list_images_in_dir(active)
                st.write(f"Found **{len(files)}** images.")
                if files:
                    st.caption("Preview (first 10):")
                    st.code("\n".join(files[:10]))

        with b2:
            st.markdown("**🖼️ Manual Bulk Upload**")
            st.caption("Or upload images directly.")

            st.markdown('<div id="clear_batch_wrap">', unsafe_allow_html=True)
            if st.button("✕ Clear all", key="clear_batch"):
                st.session_state['batch_key'] += 1
                st.session_state['b_path'] = ''
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            b_files = st.file_uploader(
                "Manual Bulk Select",
                type=["jpg", "png", "jpeg", "webp", "tiff", "tif"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key=f"bup_{st.session_state['batch_key']}"
            )

        active = st.session_state.get("b_path", "")
        valid_dir = bool(active) and os.path.isdir(active)

        if valid_dir or b_files:
            if st.button("🚀 START PIPELINE PROCESSING", type="primary", use_container_width=True):
                queue = []
                if valid_dir:
                    for f in list_images_in_dir(active):
                        queue.append({"src": os.path.join(active, f), "name": f})
                else:
                    for f in b_files:
                        queue.append({"src": f, "name": f.name})

                if queue:
                    results = []
                    prog = st.progress(0.0)
                    status = st.empty()
                    for i, item in enumerate(queue):
                        status.caption(f"Processing {item['name']} ({i+1}/{len(queue)})...")
                        try:
                            p = process_img(Image.open(item['src']), model)
                            results.append({
                                "File": item['name'],
                                "Prediction": CLASSES[int(np.argmax(p))],
                                "Confidence": f"{float(np.max(p)):.2%}"
                            })
                        except Exception as e:
                            results.append({"File": item['name'], "Prediction": "ERROR", "Confidence": str(e)})
                        prog.progress((i+1) / len(queue))
                    status.empty()

                    df = pd.DataFrame(results)
                    ra, rb = st.columns([1.5, 1])
                    with ra:
                        st.dataframe(df, use_container_width=True)
                        st.download_button(
                            "📥 DOWNLOAD FULL TELEMETRY (CSV)",
                            df.to_csv(index=False).encode("utf-8"),
                            "engine_report.csv", "text/csv",
                            use_container_width=True
                        )
                    with rb:
                        if not df.empty:
                            st.bar_chart(df["Prediction"].value_counts())
                else:
                    st.warning("No valid images found.")
        else:
            st.info("Configuration required: confirm a directory path or upload images manually.")

st.markdown('<div class="footer"><p>© 2026 Ebio-Net Systems • v3.6 Industrial Core</p></div>', unsafe_allow_html=True)
