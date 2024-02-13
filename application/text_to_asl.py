import pandas as pd
import re
import os

# absolute_path = os.path.abspath(os.path.dirname('ASL.csv'))
asl_classes = os.getcwd()+'/signs/ASL_classes.csv'

rule_id_dic = pd.read_csv(asl_classes)
classses = rule_id_dic['ENG_Class']
sign_ids = rule_id_dic['ASL_SIGN_ID']
sign_dir = rule_id_dic['ASL_SIGN_DIR']

asl_dic = {}

for i in range(len(classses)):
    asl_dic[classses[i]] = (sign_ids[i], sign_dir[i])


def get_asl_transcript(text):
    
    words = text.split()
    asl_transcript = []
    mod_text = []

    for word in words:
        asl_word = []
        mod_word = []

        for letter in word:
            if re.search("^[a-zA-Z0-9]{1}$", letter) != None:
                asl_word.append(asl_dic[letter.lower()][0])
                mod_word.append(letter)
                
        if len(mod_word) != 0: mod_text.append("".join(mod_word))   
        if len(asl_word) != 0: asl_transcript.append(asl_word)
    
    return [" ".join(mod_text), asl_transcript]
        

