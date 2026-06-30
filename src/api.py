# src/api.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from src.services.speaker_manager import SpeakerManager
from src.config import REGISTERED_DIR

app = FastAPI(title="Edge Speaker Identification API")
manager = SpeakerManager()


@app.post("/register/")
async def register_user(name: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    file_path = REGISTERED_DIR / f"{name}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    success = manager.register_speaker(name, str(file_path))

    if success:
        return {"status": "success", "message": f"User {name} registered successfully."}
    else:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail="Failed to extract embedding from audio")


@app.post("/identify/")
async def identify_user(file: UploadFile = File(...)):
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    temp_path = REGISTERED_DIR / f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    speaker_name, score = manager.identify_speaker(str(temp_path), threshold=0.6)

    if temp_path.exists():
        os.remove(temp_path)

    return {
        "status": "success",
        "identified_speaker": speaker_name,
        "confidence_score": round(score, 4)
    }