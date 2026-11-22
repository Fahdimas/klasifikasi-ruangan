import io, os
import numpy as np, pandas as pd, cv2, joblib, streamlit as st
from PIL import Image, ImageOps
from fitur import ekstrak_bgr

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hasil", "model_klasifikasi_ruangan.joblib")

st.set_page_config(page_title="Rapi atau Berantakan?", page_icon="🧹", layout="centered")


@st.cache_resource
def muat_model():
    obj = joblib.load(MODEL_PATH)
    return obj["model"], obj["meta"]


def baca_gambar(file) -> np.ndarray:
    """Foto HP -> array BGR (koreksi rotasi EXIF dulu)."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(file.getvalue()))).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


model, meta = muat_model()

st.title("🧹 Rapi atau Berantakan?")
st.caption("Klasifikasi foto ruangan berbasis fitur warna (RGB) + tekstur (GLCM)")

tab_kamera, tab_upload = st.tabs(["📷 Kamera", "🖼️ Upload foto"])
with tab_kamera:
    foto = st.camera_input("Arahkan ke ruangan, lalu ambil foto")
with tab_upload:
    unggah = st.file_uploader("Pilih foto ruangan", type=["jpg", "jpeg", "png", "webp"])

file = foto or unggah
if file is not None:
    bgr = baca_gambar(file)
    fitur = ekstrak_bgr(bgr)
    x = pd.DataFrame([fitur])[meta["fitur"]]            # urutan kolom HARUS sama dengan saat training
    proba = model.predict_proba(x)[0]
    kelas = list(model.classes_)
    label = kelas[int(np.argmax(proba))]

    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if label == "rapi":
        st.success(f"✅ RAPI  ({proba.max():.0%} yakin)")
    else:
        st.error(f"❌ BERANTAKAN  ({proba.max():.0%} yakin)")
    for k, p in zip(kelas, proba):
        st.progress(float(p), text=f"{k}: {p:.1%}")

    terang = np.mean([fitur["mean_R"], fitur["mean_G"], fitur["mean_B"]])
    if not (meta["kecerahan_min"] <= terang <= meta["kecerahan_max"]):
        st.warning("Pencahayaan foto di luar rentang data latih (terlalu gelap/terang). Hasil bisa kurang akurat, coba foto ulang.")
    if proba.max() < 0.6:
        st.info("Model kurang yakin. Coba ambil dari sudut lain yang menampakkan seluruh ruangan.")

    with st.expander("Detail fitur yang dipakai model"):
        st.dataframe(x.T.rename(columns={0: "nilai"}).round(4))

with st.expander("Info model"):
    st.write(f"**Model:** {meta['model']}  |  **Fitur:** {', '.join(meta['fitur'])}")
    st.write(f"**Data latih:** {meta['n_data']} foto  |  **CV:** {meta['cv_acc']:.1%}  |  **Test:** {meta['test_acc']:.1%}")