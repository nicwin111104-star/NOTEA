import base64
import io
import os
import re
from functools import lru_cache

import cv2
import numpy as np
import soundfile as sf
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from rapidfuzz import process
from transformers import (
    SpeechT5FeatureExtractor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
    SpeechT5Tokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

app = FastAPI(title="Notea API", description="OCR TrOCR + TTS SpeechT5", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Railway tidak punya /content/drive. Taruh folder speecht5_en di repo,
# atau set Environment Variable TTS_MODEL_DIR=/app/speecht5_en
TTS_MODEL_DIR = os.getenv("TTS_MODEL_DIR", "./speecht5_en")
OCR_MODEL_NAME = os.getenv("OCR_MODEL_NAME", "microsoft/trocr-base-handwritten")

class TtsRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"success": True, "message": "Notea API running", "endpoints": ["/health", "/ocr", "/tts"]}

@app.get("/health")
def health():
    return {"success": True, "status": "ok", "tts_model_dir_exists": os.path.isdir(TTS_MODEL_DIR)}

# =======================
# Lazy-load OCR model
# =======================
@lru_cache(maxsize=1)
def get_ocr_components():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(OCR_MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(OCR_MODEL_NAME).to(device)
    model.eval()
    return processor, model, device

# =======================
# Lazy-load TTS model
# =======================
@lru_cache(maxsize=1)
def get_tts_components():
    base_dir = TTS_MODEL_DIR
    proc_dir = os.path.join(base_dir, "processor")
    feat_dir = os.path.join(proc_dir, "feature_extractor")
    model_dir = os.path.join(base_dir, "model")
    vocoder_dir = os.path.join(base_dir, "vocoder")
    spk_path = os.path.join(base_dir, "speaker_embeddings.pt")

    missing = [p for p in [proc_dir, feat_dir, model_dir, vocoder_dir, spk_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Folder/file TTS belum ada di Railway. Upload folder speecht5_en ke repo atau set TTS_MODEL_DIR. Missing: "
            + ", ".join(missing)
        )

    tokenizer = SpeechT5Tokenizer.from_pretrained(proc_dir, local_files_only=True)
    feature_extractor = SpeechT5FeatureExtractor.from_pretrained(feat_dir, local_files_only=True)
    processor = SpeechT5Processor(tokenizer=tokenizer, feature_extractor=feature_extractor)
    model = SpeechT5ForTextToSpeech.from_pretrained(model_dir, local_files_only=True)
    vocoder = SpeechT5HifiGan.from_pretrained(vocoder_dir, local_files_only=True)
    speaker_embeddings = torch.load(spk_path, map_location="cpu")

    model.eval()
    vocoder.eval()
    return processor, model, vocoder, speaker_embeddings

# =======================
# OCR preprocessing
# =======================
def auto_crop_gray(gray_img: np.ndarray):
    _, th = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_inv = 255 - th
    ys, xs = np.where(th_inv > 0)
    if len(xs) == 0 or len(ys) == 0:
        return gray_img
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    return gray_img[y1:y2 + 1, x1:x2 + 1]

def split_lines(gray_img: np.ndarray):
    H, W = gray_img.shape
    _, th = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_inv = 255 - th
    kernel_width = max(1, int(W * 0.8))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    connected = cv2.morphologyEx(th_inv, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_imgs = []
    min_h = max(8, int(0.015 * H))
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < min_h:
            continue
        roi = gray_img[y:y + h, :]
        line_imgs.append((y, Image.fromarray(roi).convert("RGB")))

    line_imgs = [img for y, img in sorted(line_imgs, key=lambda z: z[0])]
    if not line_imgs:
        line_imgs = [Image.fromarray(gray_img).convert("RGB")]
    return line_imgs

def clean_line(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^[^A-Za-z0-9]+", "", t)
    t = re.sub(r"^[0-9]+\s*[\.)]\s*", "", t)
    t = re.sub(r"^#\s*", "", t)
    m = re.match(r"^([A-Za-z])\s+(.+)$", t)
    if m and len(m.group(2)) >= 5:
        t = m.group(2)
    t = re.sub(r"\b\d{3,4}\b", "", t)
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    t = re.sub(r"([,.!?])([A-Za-z])", r"\1 \2", t)
    t = re.sub(r"\s+", " ", t)
    if len(t) <= 2 and not re.search(r"[A-Za-z]{2,}", t):
        return ""
    if re.fullmatch(r"[0-9 ]+", t):
        return ""
    if re.fullmatch(r"[\W_]+", t):
        return ""
    return t.strip(" ,")

def ocr_pipeline(pil_img: Image.Image) -> str:
    ocr_processor, ocr_model, device_ocr = get_ocr_components()
    gray = np.array(pil_img.convert("L"))
    gray_cropped = auto_crop_gray(gray)
    lines = split_lines(gray_cropped)

    texts = []
    for line_img in lines:
        w, h = line_img.size
        line_resized = line_img.resize((int(w * 2.0), int(h * 2.0)))
        inputs = ocr_processor(images=line_resized, return_tensors="pt").to(device_ocr)
        with torch.no_grad():
            ids = ocr_model.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        t = ocr_processor.batch_decode(ids, skip_special_tokens=True)[0]
        texts.append(t)

    lines_clean = [clean_line(t) for t in texts]
    lines_clean = [t for t in lines_clean if t]
    if lines_clean:
        m = re.match(r"^[aA]\s+([A-Z].+)", lines_clean[0])
        if m:
            lines_clean[0] = m.group(1)
    return "\n".join(lines_clean).strip()

# =======================
# Confidence OCR
# =======================
VOCAB = [
    "hello", "hi", "hey", "good", "morning", "afternoon", "evening", "night",
    "thanks", "thank", "you", "sorry", "please", "halo", "hai", "selamat",
    "pagi", "siang", "sore", "malam", "terima", "kasih", "maaf", "tolong",
    "i", "me", "my", "your", "student", "college", "school", "university",
    "mahasiswa", "pelajar", "sekolah", "kampus", "belajar", "kuliah", "baca", "tulis",
]

def line_similarity(line: str) -> float:
    words = re.findall(r"[A-Za-z']+", line.lower())
    if not words:
        return 0.0
    scores = []
    for w in words:
        match = process.extractOne(w, VOCAB)
        scores.append((match[1] / 100.0) if match else 0.0)
    return float(np.mean(scores)) if scores else 0.0

def line_noise(line: str) -> float:
    if not line:
        return 1.0
    total = len(line)
    strange = len(re.findall(r"[^A-Za-z0-9\s,.!?']", line))
    return min(1.0, strange / max(total, 1))

similarity_var = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "similarity")
noise_var = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "noise")
trust_var = ctrl.Consequent(np.arange(0, 1.01, 0.01), "trust")
similarity_var["low"] = fuzz.trimf(similarity_var.universe, [0.0, 0.0, 0.4])
similarity_var["med"] = fuzz.trimf(similarity_var.universe, [0.2, 0.5, 0.8])
similarity_var["high"] = fuzz.trimf(similarity_var.universe, [0.6, 1.0, 1.0])
noise_var["low"] = fuzz.trimf(noise_var.universe, [0.0, 0.0, 0.3])
noise_var["high"] = fuzz.trimf(noise_var.universe, [0.2, 1.0, 1.0])
trust_var["low"] = fuzz.trimf(trust_var.universe, [0.0, 0.0, 0.4])
trust_var["med"] = fuzz.trimf(trust_var.universe, [0.2, 0.5, 0.8])
trust_var["high"] = fuzz.trimf(trust_var.universe, [0.6, 1.0, 1.0])
trust_ctrl = ctrl.ControlSystem([
    ctrl.Rule(similarity_var["high"] & noise_var["low"], trust_var["high"]),
    ctrl.Rule(similarity_var["med"] & noise_var["low"], trust_var["med"]),
    ctrl.Rule(similarity_var["low"] | noise_var["high"], trust_var["low"]),
])

def fuzzy_trust_for_line(line: str) -> float:
    sim_ctrl = ctrl.ControlSystemSimulation(trust_ctrl)
    sim_ctrl.input["similarity"] = line_similarity(line)
    sim_ctrl.input["noise"] = line_noise(line)
    sim_ctrl.compute()
    return float(sim_ctrl.output["trust"])

def ocr_confidence(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    scores = [fuzzy_trust_for_line(ln) for ln in lines]
    return float(sum(scores) / len(scores))

# =======================
# TTS
# =======================
def run_tts(text: str) -> bytes:
    if not text.strip():
        raise ValueError("Text untuk TTS kosong.")
    processor, model, vocoder, speaker_embeddings = get_tts_components()
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embeddings, vocoder=vocoder)
    buf = io.BytesIO()
    sf.write(buf, speech.cpu().numpy(), 16000, format="WAV")
    return buf.getvalue()

@app.post("/ocr")
async def ocr_api(file: UploadFile = File(...)):
    return {
        "success": False,
        "error": "OCR sementara dinonaktifkan untuk test TTS di Railway."
    }

"""@app.post("/ocr")
async def ocr_api(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        text = ocr_pipeline(pil_img)
        return {"success": True, "text": text, "confidence": ocr_confidence(text) if text.strip() else 0.0}
    except Exception as e:
        return {"success": False, "error": str(e)}"""

@app.post("/tts")
def tts_api(req: TtsRequest):
    try:
        audio_bytes = run_tts(req.text)
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="tts_output.wav"'
            }
        )
    except Exception as e:
        return {"success": False, "error": str(e)}

"""@app.post("/tts")
def tts_api(req: TtsRequest):
    return {
        "success": False,
        "error": "TTS sementara dinonaktifkan di server Railway agar OCR lebih stabil."
    }"""