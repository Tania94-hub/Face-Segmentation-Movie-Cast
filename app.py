
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K
import time
import pandas as pd
from PIL import Image
import io

# ── Custom functions ─────────────────────────────────────────
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return dice_loss(y_true, y_pred) + bce

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "vgg16_unet_best.h5",
        custom_objects={
            "combined_loss": combined_loss,
            "dice_coefficient": dice_coefficient,
            "iou_metric": iou_metric
        }
    )

def preprocess(pil_image):
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, (256, 256))
    return img / 255.0

def overlay_boxes(original_pil, pred_mask, threshold=0.5):
    img = cv2.resize(np.array(original_pil.convert("RGB")),
                     (pred_mask.shape[1], pred_mask.shape[0]))
    binary = (pred_mask > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = img.copy()
    face_count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 300:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result, f"Face {face_count+1}", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            face_count += 1
    overlay = result.copy()
    overlay[binary == 1] = [0, 200, 0]
    blended = cv2.addWeighted(result, 0.75, overlay, 0.25, 0)
    return blended, binary, face_count

# ── App Layout ───────────────────────────────────────────────
st.set_page_config(page_title="Scene Cast AI", layout="wide", page_icon="🎬")

st.title("🎬 Scene Cast AI")
st.markdown("### Real-Time Face Segmentation for Movie Cast Identification")
st.markdown("Upload a movie scene image to detect and segment faces instantly.")
st.divider()

model = load_model()

col_upload, col_settings = st.columns([3, 1])
with col_upload:
    uploaded = st.file_uploader("Upload Movie Scene Image", type=["jpg","jpeg","png"])
with col_settings:
    threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)
    st.caption("Lower = more sensitive, Higher = more precise")

if uploaded:
    pil_image = Image.open(uploaded)

    col1, col2, col3 = st.columns(3)
    col1.subheader("Original")
    col1.image(pil_image, use_column_width=True)

    with st.spinner("Detecting faces..."):
        img_array = preprocess(pil_image)
        img_batch = np.expand_dims(img_array, 0)
        start     = time.time()
        pred_mask = model.predict(img_batch, verbose=0)[0, :, :, 0]
        infer_ms  = (time.time() - start) * 1000

    result_img, binary_mask, face_count = overlay_boxes(pil_image, pred_mask, threshold)
    face_coverage = binary_mask.sum() / binary_mask.size * 100

    col2.subheader("Predicted Mask")
    col2.image((pred_mask * 255).astype(np.uint8), use_column_width=True, clamp=True)

    col3.subheader("Detected Faces")
    col3.image(result_img, use_column_width=True)

    st.divider()
    st.subheader("📊 Performance Dashboard")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Faces Detected", face_count)
    m2.metric("Inference Speed", f"{infer_ms:.1f} ms")
    m3.metric("Face Coverage", f"{face_coverage:.1f}%")
    m4.metric("Threshold", threshold)

    st.divider()
    dl1, dl2 = st.columns(2)

    # Download log
    log_df = pd.DataFrame({
        "Metric": ["Faces Detected","Inference Time (ms)","Face Coverage (%)","Threshold"],
        "Value":  [face_count, round(infer_ms,2), round(face_coverage,2), threshold]
    })
    csv_buf = io.StringIO()
    log_df.to_csv(csv_buf, index=False)
    dl1.download_button("⬇ Download Detection Log",
                        data=csv_buf.getvalue(),
                        file_name="detection_log.csv",
                        mime="text/csv")

    # Download result image
    result_pil = Image.fromarray(result_img)
    img_buf = io.BytesIO()
    result_pil.save(img_buf, format="PNG")
    dl2.download_button("⬇ Download Result Image",
                        data=img_buf.getvalue(),
                        file_name="detected_faces.png",
                        mime="image/png")

    st.divider()
    st.subheader("📈 Model Performance Summary")
    perf_df = pd.DataFrame({
        "Model":        ["MobileNetV2 U-Net","Custom U-Net","VGG16 U-Net","VGG16 Fine-tuned"],
        "Val Dice":     [0.679, 0.594, 0.672, 0.652],
        "Val IoU":      [0.517, 0.426, 0.466, 0.470],
        "Augmentation": ["None","Basic","Albumentations 3x","Albumentations 3x"]
    })
    st.dataframe(perf_df, use_container_width=True)

st.markdown("---")
st.caption("Built with TensorFlow + Streamlit | GUVI HCL Final Project — Tania Banerjee")
