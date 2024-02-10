from fastapi import FastAPI
import torch
import torchaudio
import sys
sys.path.insert(1, '/application/')
from application.asl_generator import getASL

app = FastAPI()

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
api_test_audios  = ["C:/Users/adria/Desktop/SpeechToASL_API/data/84-121123-0000.flac", "C:/Users/adria/Desktop/SpeechToASL_API/data/84-121123-0001.flac", "C:/Users/adria/Desktop/SpeechToASL_API/data/84-121123-0002.flac", "C:/Users/adria/Desktop/SpeechToASL_API/data/84-121123-0003.flac"]

print('Speech to ASL API [STARTED]')
print('Speech to ASL API [STARTED]')
print('Speech to ASL API [STARTED]')

@app.get("/")
def read_root():
    sign_filenames = getASL(device, model, bundle, api_test_audios[0])
    return {"sign_image_paths": sign_filenames}

@app.get("/audio/{item_id}")
def read_item(item_id: int, query_param: str = None):
    if item_id > len(api_test_audios): return {"NULL": None}
    return {"sign_image_paths": getASL(device, model, bundle, api_test_audios[item_id])}
