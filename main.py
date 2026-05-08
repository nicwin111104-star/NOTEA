import os
import tempfile

import edge_tts
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(
    title="Notea API",
    description="Notea API - OCR + Edge TTS",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TtsRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "success": True,
        "message": "Notea API running",
        "endpoints": ["/health", "/ocr", "/tts"],
    }


@app.get("/health")
def health():
    return {
        "success": True,
        "status": "ok",
        "tts": "edge-tts",
    }


@app.post("/ocr")
async def ocr_api(file: UploadFile = File(...)):
    return {
        "success": False,
        "error": "OCR sementara dinonaktifkan untuk test Edge TTS di Railway.",
    }


@app.post("/tts")
async def tts_api(req: TtsRequest):
    try:
        text = req.text.strip()

        if not text:
            return {
                "success": False,
                "error": "Text kosong.",
            }

        output_path = os.path.join(
            tempfile.gettempdir(),
            "tts_output.mp3",
        )

        communicate = edge_tts.Communicate(
            text,
            voice="id-ID-ArdiNeural",
        )

        await communicate.save(output_path)

        return FileResponse(
            output_path,
            media_type="audio/mpeg",
            filename="tts_output.mp3",
        )

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }