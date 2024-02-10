import sys
sys.path.insert(2, '/application/')
from .data_scripts import custom_audio_input
from .greedyCTCDecoder import GreedyCTCDecoder
from .text_to_asl import get_asl_transcript
import torch
from matplotlib import pyplot as plt
import pandas as pd
import IPython
import torchaudio
from PIL import Image
import os


def getASL(device, model, bundle, audio):
    # data_scripts.custom_audio_input returns audio id, loation, transcript, tensor, and sample rate
    train_data_info = custom_audio_input(audio, "ph-remove in future")

    waveform = train_data_info[0]['tensor']
    sample_rate = train_data_info[0]['sample_rate']
    waveform.to(device)

    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

    #  extract acoustic features from the audio.
    with torch.inference_mode():
        features, _ = model.extract_features(waveform)
        
    # feature extraction and classification
    with torch.inference_mode():
        emission, _ = model(waveform)

    decoder = GreedyCTCDecoder(labels=bundle.get_labels())
    generated_transcript = decoder(emission[0])

    generate_transcript_mod = " ".join(generated_transcript.split('|'))
    generated_transcript_sign_ids = get_asl_transcript(generate_transcript_mod)

    rule_id_dic = pd.read_csv('C:/Users/adria/Desktop/SpeechToASL_API/data/ASL_classes.csv')
    classses = rule_id_dic['ENG_Class']
    sign_ids = rule_id_dic['ASL_SIGN_ID']
    sign_dir = rule_id_dic['ASL_SIGN_DIR']

    asl_dic = {}

    for i in range(len(classses)):
        asl_dic[sign_ids[i]] = sign_dir[i]

    # Store the sign directories associated to sign ids
    asl_sign_paths = []

    for asl in generated_transcript_sign_ids[1]:
        word = []
        
        for letter in asl:
            word.append(asl_dic[letter])
            
        asl_sign_paths.append(word)

    # GENERATE ASL IMAGES OF TEXT
    words = 1
    letter = 0

    merged_asl = None
    folder = '/Users/adria/Desktop/SpeechToASL_API/output/'
    signs_folder = '/Users/adria/Desktop/SpeechToASL_API/signs/'

    for word_paths in asl_sign_paths:
        letters = len(word_paths)
        pos_indx = 1
        first_image = Image.open(signs_folder+word_paths[0])
        merge = Image.new('RGB', (letters*first_image.size[0], first_image.size[1]), (250, 250, 250))
        merge.paste(first_image, (0,0))
        
        for i in range(1, letters):
            merge.paste(Image.open(signs_folder+word_paths[i]), (pos_indx*first_image.size[0],0))
            pos_indx +=1
            
        merge.save(folder+'word_' + str(words) + '.jpg', 'jpeg')
        words+=1
    

    return asl_sign_paths