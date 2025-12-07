import os
from collections import Counter
import io
import zipfile

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import altair as alt

# ------------------ CONFIG ------------------

# Local path on your laptop
LOCAL_MODEL_PATH = r"C:\Users\asus\OneDrive\Desktop\yolo deploy\best.pt"
# Path used on Streamlit Cloud (best.pt in same folder as app.py)
CLOUD_MODEL_PATH = "best.pt"

# Automatically pick local path if it exists, else use cloud path
MODEL_PATH = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else CLOUD_MODEL_PATH

CONFIDENCE = 0.25
IOU = 0.45

st.set_page_config(
    page_title="CircuitGuard – PCB Defect Detection",
    page_icon="🛡️",
    layout="wide"
)

# ------------------ CUSTOM STYLING ------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Bitcount+Prop+Single:wght@400;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background: #f8fbff;
        font-family: 'Poppins', sans-serif;
        color: #102a43;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e8f5ff 0%, #e7fff7 100%);
        border-right: 1px solid #d0e2ff;
    }

    /* Make sidebar text dark so it's readable */
    [data-testid="stSidebar"] * {
        color: #102a43 !important;
    }

    /* Code block for best.pt: light background, dark text */
    [data-testid="stSidebar"] pre, [data-testid="stSidebar"] code {
        background: #e5e7eb !important;
        color: #111827 !important;
    }

    /* Top toolbar (Share, etc.) */
    [data-testid="stToolbar"] * {
        color: #e5e7eb !important;
    }

    h2, h3 {
        font-weight: 600;
        color: #13406b;
    }

    .stButton>button {
        border-radius: 999px;
        padding: 0.5rem 1.25rem;
        border: none;
        font-weight: 500;
        background: #85c5ff;
        color: #0f172a;
    }

    .stButton>button:hover {
        background: #63b1ff;
    }

    .upload-box {
        border-radius: 18px;
        border: 1px dashed #a3c9ff;
        padding: 1.5rem;
        background: #ffffff;
    }

    /* File uploader text & Browse button */
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] label {
        color: #f9fafb !important;  /* light text on dark drop area */
    }
    [data-testid="stFileUploader"] button {
        background: #111827 !important;
        color: #f9fafb !important;
        border-radius: 999px !important;
        border: none !important;
    }

    .metric-card {
        border-radius: 18px;
        padding: 0.75rem 1rem;
        background: #ffffff;
        border: 1px solid #dbeafe;
    }

    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
        margin-bottom: 0.1rem;
    }

    .metric-value {
        font-size: 1.15rem;
        font-weight: 600;
        color: #111827;
    }

    .logo-circle {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: #e0f2fe;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 32px;
        margin-bottom: 0.4rem;
    }

    .header-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
    }

    .main-title {
        font-family: 'Bitcount Prop Single', system-ui, -apple-system,
                     BlinkMacSystemFont, 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 2.8rem;
        text-align: center;
        color: #13406b;
    }

    .subtitle-text {
        font-size: 0.95rem;
        color: #334e68;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ SESSION STATE ------------------
if "full_results_df" not in st.session_state:
    st.session_state["full_results_df"] = None
if "annotated_images" not in st.session_state:
    st.session_state["annotated_images"] = []
if "show_download" not in st.session_state:
    st.session_state["show_download"] = False

# ------------------ MODEL LOADING & INFERENCE ------------------
@st.cache_resource
def load_model(path: str):
    """Load YOLO model once and cache it."""
    return YOLO(path)


def run_inference(model, image):
    """Run detection and return plotted image + raw result."""
    results = model.predict(image, conf=CONFIDENCE, iou=IOU)
    r = results[0]
    plotted = r.plot()  # BGR numpy array
    plotted = plotted[:, :, ::-1]  # BGR -> RGB
    pil_img = Image.fromarray(plotted)
    return pil_img, r


def get_class_counts(result, class_names):
    """Return a dict: {class_name: count} for one result."""
    if len(result.boxes) == 0:
        return {}
    cls_indices = result.boxes.cls.tolist()
    labels = [class_names[int(i)] for i in cls_indices]
    counts = Counter(labels)
    return dict(counts)


def get_defect_locations(result, class_names, image_name):
    """Return rows with defect type, confidence and bounding box coords + image name."""
    if len(result.boxes) == 0:
        return []

    boxes = result.boxes
    xyxy = boxes.xyxy.tolist()
    cls_indices = boxes.cls.tolist()
    confs = boxes.conf.tolist()

    rows = []
    for coords, c, cf in zip(xyxy, cls_indices, confs):
        x1, y1, x2, y2 = coords
        rows.append({
            "Image": image_name,
            "Defect type": class_names[int(c)],
            "Confidence": round(float(cf), 2),
            "x1": round(float(x1), 1),
            "y1": round(float(y1), 1),
            "x2": round(float(x2), 1),
            "y2": round(float(y2), 1),
        })

    return rows


# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.subheader("Model configuration")
    st.write("**Active model path:**")
    st.code(MODEL_PATH, language="text")

    st.markdown("----")
    st.subheader("Model performance")
    st.markdown(
        """
        **mAP@50:** 0.9823  
        **mAP@50–95:** 0.5598  
        **Precision:** 0.9714  
        **Recall:** 0.9765
        """
    )

# ------------------ MAIN LAYOUT ------------------
# Logo + main heading (centered, big)
st.markdown(
    """
    <div class="header-container">
        <div class="logo-circle">🛡️</div>
        <div class="main-title">CircuitGuard – PCB Defect Detection</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top metrics row with custom cards
metric_cols = st.columns(4)
metric_info = [
    ("mAP@50", "0.9823"),
    ("mAP@50–95", "0.5598"),
    ("Precision", "0.9714"),
    ("Recall", "0.9765"),
]
for col, (label, value) in zip(metric_cols, metric_info):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <p class="subtitle-text">
    Detect and highlight <strong>PCB defects</strong> such as missing hole, mouse bite,
    open circuit, short, spur and spurious copper using a YOLO-based deep learning model.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Upload PCB Images")

with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload one or more PCB images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ DETECTION & DISPLAY ------------------
if uploaded_files:
    try:
        model = load_model(MODEL_PATH)
        class_names = model.names  # dict: {id: name}
    except Exception as e:
        st.error(f"Error loading model from `{MODEL_PATH}`: {e}")
    else:
        global_counts = Counter()
        all_rows = []
        annotated_images = []

        for file in uploaded_files:
            st.markdown(f"#### 📷 {file.name}")
            img = Image.open(file).convert("RGB")

            with st.spinner("Running detection..."):
                plotted_img, result = run_inference(model, img)

            # Show detection image (smaller)
            st.image(
                plotted_img,
                caption="Detections",
                width=550,
            )

            # Keep annotated image for download later
            annotated_images.append((file.name, plotted_img))

            # Summary text
            if len(result.boxes) == 0:
                st.success("No defects detected in this image.")
            else:
                st.info(f"Detected **{len(result.boxes)}** potential defect(s).")

                # Update global defect counts
                counts = get_class_counts(result, class_names)
                global_counts.update(counts)

                # Defect locations for this image (table on screen)
                loc_rows = get_defect_locations(result, class_names, file.name)
                all_rows.extend(loc_rows)

                if loc_rows:
                    loc_df = pd.DataFrame(loc_rows)
                    st.markdown("**Defect locations (bounding boxes in pixels):**")
                    st.dataframe(loc_df.drop(columns=["Image"]), use_container_width=True)

            st.markdown("---")

        # Build full results DF for export (all images)
        if all_rows:
            full_results_df = pd.DataFrame(all_rows)
            st.session_state["full_results_df"] = full_results_df
            st.session_state["annotated_images"] = annotated_images
        else:
            st.session_state["full_results_df"] = None
            st.session_state["annotated_images"] = []

        # After processing all images: single combined bar chart
        if sum(global_counts.values()) > 0:
            st.subheader("Overall defect distribution across all uploaded images")
            global_df = pd.DataFrame(
                {"Defect Type": list(global_counts.keys()),
                 "Count": list(global_counts.values())}
            )

            chart = (
                alt.Chart(global_df)
                .mark_bar(size=40)
                .encode(
                    x=alt.X("Defect Type:N",
                            sort="-y",
                            axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Count:Q"),
                    tooltip=["Defect Type", "Count"],
                )
                .properties(height=350)
            )

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No defects detected in any of the uploaded images.")

        # -------- Export flow: Finish + Download (CSV + annotated images) --------
        if st.session_state["full_results_df"] is not None:
            st.markdown("### Export results")
            if st.button("Finish defect detection"):
                st.session_state["show_download"] = True

            if st.session_state["show_download"]:
                # Build ZIP (CSV + annotated images)
                full_results_df = st.session_state["full_results_df"]
                annotated_images = st.session_state["annotated_images"]

                csv_bytes = full_results_df.to_csv(index=False).encode("utf-8")

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    # CSV
                    zf.writestr("circuitguard_detection_results.csv", csv_bytes)
                    # Annotated images
                    for name, pil_img in annotated_images:
                        img_bytes_io = io.BytesIO()
                        pil_img.save(img_bytes_io, format="PNG")
                        img_bytes_io.seek(0)
                        base = os.path.splitext(name)[0]
                        zf.writestr(f"annotated_{base}.png", img_bytes_io.getvalue())

                zip_buffer.seek(0)

                st.download_button(
                    "Download results (CSV + annotated images, ZIP)",
                    data=zip_buffer,
                    file_name="circuitguard_results.zip",
                    mime="application/zip",
                )
else:
    st.info("Upload one or more PCB images to start detection.")
