from fastapi import FastAPI
import torch
import torchaudio
import os
from application.asl_generator import getASL
from fastapi import FastAPI, File, UploadFile, Request
import shutil
import re

app = FastAPI()


torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
audio_id_queue  = {}


@app.get("/")
async def read_root():
    return {"STATUS": 'API WORKING'}

@app.get("/asl_images/{audio_id}")
def get_asl_images(audio_id: str, query_param: str = None):
    if audio_id not in audio_id_queue.keys(): return {"NULL": None}
    audio = audio_id_queue.get(audio_id)
    audio_fn = audio.split('/')
    audio_fn = audio_fn[len(audio_fn)-1].split('.')[0]

    # TASK: Search for audio_fn in outputs and return list of words in folder - RETHINK TASK, ON IMPLEMENTATION AUDIO FILE NAMES WILL BE DIFFERENR
    
    return {"sign_image_paths": getASL(device, model, bundle, audio, audio_fn)}


@app.post("/upload_audio")
async def upload_audio(request: Request):
    """
    Example POST request endpoint that receives an audio file.
    """
    return request
    # audio_file = os.getcwd() + f"/inputs/{file.filename}"
    # audio_fn = file.filename.split('.')[0]
    # audio_id_queue[audio_fn] = audio_file


    # with open(audio_file, 'w+b') as file2:
    #     shutil.copyfileobj(file.file, file2)

    # return {
    #     'audio_id': audio_fn,
    #     'file': file.filename    
    # }