Cara pakai di Railway:

1. Buat folder project berisi:
   - main.py
   - requirements.txt
   - Procfile
   - speecht5_en/  (folder model TTS dari Google Drive kamu)

2. Struktur folder TTS harus seperti ini:
   speecht5_en/
     processor/
       tokenizer files...
       feature_extractor/
     model/
     vocoder/
     speaker_embeddings.pt

3. Push ke GitHub.
4. Railway -> New Project -> Deploy from GitHub.
5. Start command:
   uvicorn main:app --host 0.0.0.0 --port $PORT

6. Test endpoint:
   GET  https://domain-railway/health
   POST https://domain-railway/ocr  form-data key=file
   POST https://domain-railway/tts  JSON {"text":"Halo ini tes Notea"}

Catatan penting:
- Jangan pakai google.colab, drive.mount, ngrok, atau path /content/drive di Railway.
- Kalau deploy gagal karena memori, pisahkan OCR dan TTS jadi dua service.
- Token ngrok yang ada di notebook lama sebaiknya di-reset/rotate.
