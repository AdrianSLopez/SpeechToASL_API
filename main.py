from fastapi import FastAPI
import torch
import torchaudio
import sys
sys.path.insert(1, '/application/')
from application.asl_generator import getASL
from application.models.AudioModel import AudioModel
from fastapi import FastAPI, File, UploadFile
import shutil

app = FastAPI()

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
audio_id_queue  = {
    0: "C:/Users/adria/Desktop/SpeechToASL_API/data/84-121123-0000.flac", 
    1: "C:/Users/adria/Desktop/SpeechToASL_API/data/84-121123-0001.flac", 
    2: "C:/Users/adria/Desktop/SpeechToASL_API/data/84-121123-0002.flac", 
    3: "C:/Users/adria/Desktop/SpeechToASL_API/data/84-121123-0003.flac"}

print('Speech to ASL API [STARTED]')
print('Speech to ASL API [STARTED]')
print('Speech to ASL API [STARTED]')

@app.get("/")
async def read_root():
    sign_filenames = getASL(device, model, bundle, audio_id_queue[0])
    return {"sign_image_paths": '2'}

@app.get("/asl_images/{audio_id}")
def get_asl_images(audio_id: int, query_param: str = None):
    if audio_id >= len(audio_id_queue): return {"NULL": None}
    return {"sign_image_paths": getASL(device, model, bundle, audio_id_queue[audio_id])}


@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """
    Example POST request endpoint that receives an audio file.
    """
    path = f"files/{file.filename}"
    audio_id_queue[len(audio_id_queue)] = "C:/Users/adria/Desktop/SpeechToASL_API/files/"+file.filename 


    with open(path, 'w+b') as file2:
        shutil.copyfileobj(file.file, file2)

    return {
        'audio_id': len(audio_id_queue)-1,
        'file': file.filename    
    }