"""Ekstraksi 14 fitur (9 warna RGB + 5 GLCM). Dipakai bareng oleh notebook DAN app Streamlit
supaya fitur saat training & saat prediksi dihitung dengan cara yang SAMA persis."""
import numpy as np, cv2
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops

SIZE  = 256
COLOR = [f"{a}_{c}" for a in ("mean", "std", "skew") for c in "RGB"]
TEX   = ["energi", "entropy", "kontras", "korelasi", "homogenitas"]
SEMUA = COLOR + TEX


def glcm_feats(gray, distances=(1,), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    g = (gray // 4).astype(np.uint8)                      # 256 level -> 64 level
    P = graycomatrix(g, list(distances), list(angles), levels=64, symmetric=True, normed=True)
    energy   = graycoprops(P, "energy").mean()
    contrast = graycoprops(P, "contrast").mean()
    corr     = graycoprops(P, "correlation").mean()
    homog    = graycoprops(P, "homogeneity").mean()
    p = P.mean(axis=(2, 3)); p = p[p > 0]
    entropy = -(p * np.log2(p)).sum()
    return energy, entropy, contrast, corr, homog


def ekstrak_bgr(bgr):
    """bgr = array OpenCV (H,W,3 uint8). Return dict 14 fitur."""
    bgr = cv2.resize(bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    r, g, b = [bgr[:, :, i].astype(float).ravel() for i in (2, 1, 0)]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    en, ent, con, cor, hom = glcm_feats(gray)
    d = dict(mean_R=r.mean(), mean_G=g.mean(), mean_B=b.mean(),
             std_R=r.std(), std_G=g.std(), std_B=b.std(),
             skew_R=skew(r), skew_G=skew(g), skew_B=skew(b),
             energi=en, entropy=ent, kontras=con, korelasi=cor, homogenitas=hom)
    return {k: float(np.nan_to_num(v)) for k, v in d.items()}   # skew bisa NaN kalau citra polos


def ekstrak(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"Gagal baca gambar: {path}")
    return ekstrak_bgr(bgr)