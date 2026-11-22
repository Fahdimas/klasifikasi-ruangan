import io, os, time, threading
from collections import deque

import av
import numpy as np, pandas as pd, cv2, joblib, streamlit as st
from PIL import Image, ImageOps
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from fitur import ekstrak_bgr

BASE = os.path.dirname(__file__)
# cari model di folder "hasil/" dulu, kalau tidak ada coba di folder yang sama dengan app.py
KANDIDAT = [os.path.join(BASE, "hasil", "model_klasifikasi_ruangan.joblib"),
            os.path.join(BASE, "model_klasifikasi_ruangan.joblib")]
MODEL_PATH = next((p for p in KANDIDAT if os.path.exists(p)), KANDIDAT[0])

st.set_page_config(page_title="Rapi atau Berantakan?", page_icon="🧹", layout="centered")


@st.cache_resource
def muat_model():
    obj = joblib.load(MODEL_PATH)
    return obj["model"], obj["meta"]


model, meta = muat_model()
KELAS = list(model.classes_)

HIJAU, MERAH, KUNING = (60, 200, 60), (60, 60, 230), (0, 200, 255)   # BGR


def prediksi(bgr):
    fitur = ekstrak_bgr(bgr)
    x = pd.DataFrame([fitur])[meta["fitur"]]          # urutan kolom HARUS sama dengan saat training
    proba = model.predict_proba(x)[0]
    terang = np.mean([fitur["mean_R"], fitur["mean_G"], fitur["mean_B"]])
    return proba, terang


class ProsesorRuangan(VideoProcessorBase):
    """Dipanggil untuk SETIAP frame kamera. Prediksi dijalankan tiap ~0.4 detik
    (biar tidak berat), hasilnya dihaluskan dengan rata-rata beberapa prediksi terakhir."""

    def __init__(self):
        self.riwayat = deque(maxlen=5)
        self.terakhir = 0.0
        self.label, self.conf, self.terang_ok = None, 0.0, True
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        if now - self.terakhir > 0.4:
            self.terakhir = now
            try:
                proba, terang = prediksi(img)
                self.riwayat.append(proba)
                rata = np.mean(self.riwayat, axis=0)
                with self.lock:
                    self.label = KELAS[int(np.argmax(rata))]
                    self.conf = float(rata.max())
                    self.terang_ok = meta["kecerahan_min"] <= terang <= meta["kecerahan_max"]
            except Exception:
                pass

        with self.lock:
            label, conf, terang_ok = self.label, self.conf, self.terang_ok

        h, w = img.shape[:2]
        if label is None:
            teks, warna = "Menganalisis...", KUNING
        elif conf < 0.6:
            teks, warna = f"RAGU-RAGU ({conf:.0%})", KUNING
        elif label == "rapi":
            teks, warna = f"RAPI ({conf:.0%})", HIJAU
        else:
            teks, warna = f"BERANTAKAN ({conf:.0%})", MERAH

        # kotak di sekeliling area pandang + banner label di atasnya
        m = int(min(h, w) * 0.04)
        tebal = max(3, int(min(h, w) * 0.01))
        cv2.rectangle(img, (m, m), (w - m, h - m), warna, tebal)

        skala = max(0.7, w / 640 * 0.9)
        (tw, th), _ = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, skala, 2)
        cv2.rectangle(img, (m, m), (m + tw + 20, m + th + 22), warna, -1)
        cv2.putText(img, teks, (m + 10, m + th + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    skala, (255, 255, 255), 2, cv2.LINE_AA)

        if not terang_ok:
            cv2.putText(img, "Pencahayaan di luar rentang data latih",
                        (m + 10, h - m - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.5, skala * 0.6), KUNING, 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def baca_gambar(file) -> np.ndarray:
    """Foto HP -> array BGR (koreksi rotasi EXIF dulu)."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(file.getvalue()))).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


st.title("🧹 Rapi atau Berantakan?")
st.caption("Klasifikasi ruangan berbasis fitur warna (RGB) + tekstur (GLCM)")

tab_live, tab_upload = st.tabs(["🎥 Kamera Live", "🖼️ Upload foto"])

with tab_live:
    st.write("Klik **START**, izinkan kamera, lalu arahkan ke ruangan. Hasil muncul langsung di dalam kotak.")
    webrtc_streamer(
        key="ruangan-live",
        video_processor_factory=ProsesorRuangan,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={
            "video": {"facingMode": "environment", "width": {"ideal": 640}},  # kamera belakang di HP
            "audio": False,
        },
        async_processing=True,
    )

with tab_upload:
    unggah = st.file_uploader("Pilih foto ruangan", type=["jpg", "jpeg", "png", "webp"])
    if unggah is not None:
        bgr = baca_gambar(unggah)
        proba, terang = prediksi(bgr)
        label = KELAS[int(np.argmax(proba))]

        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if label == "rapi":
            st.success(f"✅ RAPI  ({proba.max():.0%} yakin)")
        else:
            st.error(f"❌ BERANTAKAN  ({proba.max():.0%} yakin)")
        for k, p in zip(KELAS, proba):
            st.progress(float(p), text=f"{k}: {p:.1%}")

        if not (meta["kecerahan_min"] <= terang <= meta["kecerahan_max"]):
            st.warning("Pencahayaan foto di luar rentang data latih. Hasil bisa kurang akurat.")
        if proba.max() < 0.6:
            st.info("Model kurang yakin. Coba ambil dari sudut lain yang menampakkan seluruh ruangan.")

with st.expander("Info model"):
    st.write(f"**Model:** {meta['model']}  |  **Fitur:** {', '.join(meta['fitur'])}")
    st.write(f"**Data latih:** {meta['n_data']} foto  |  **CV:** {meta['cv_acc']:.1%}  |  **Test:** {meta['test_acc']:.1%}")