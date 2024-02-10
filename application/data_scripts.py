import os
import shutil
import re
import numpy as np
import torchaudio

def get_directory_info(location):
    directory_info = []
    readers = os.listdir(location)
    
    for reader in readers:
        chapters = os.listdir(location + '/' + reader)

        for chapter in chapters:
            files = os.listdir(location + reader + '/' + chapter)
            transcript = create_transcript_dictionary(location  + reader + '/' + chapter + '/' + files[len(files)-1])
            
            for file in files:
                if re.search(r'.flac$', file):
                    filename_id = re.split(r'.flac', file)[0]
                    audio_loc = location + reader + '/' + chapter + '/' + file
                    y, sr = torchaudio.load(audio_loc)

                    audio_info = {
                        "id": filename_id, 
                        "audio_loc": audio_loc,
                        "transcript": transcript.get(filename_id),
                        "tensor": y,
                        "sample_rate": sr
                    }

                    directory_info.append(audio_info)

    return directory_info

def custom_audio_input(location, transcript):
    audio_info = {
        "id": "0000",
        "audio_loc": location,
        "transcript": transcript,
    }

    y, sample_rate = torchaudio.load(location)

    audio_info['tensor'] = y
    audio_info['sample_rate'] = sample_rate

    return [audio_info]
    
def create_transcript_dictionary(transcript_file):
    transcript_dictionary = {}
    transcript = open(transcript_file)
    lines = transcript.readlines()

    for l in lines:
        line_info = l.split(' ', 1)
        transcript_dictionary[line_info[0]] = line_info[1].strip()
    
    return transcript_dictionary

def resetOuput(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder)

    os.mkdir(folder)
