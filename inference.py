import logging
import os
import pickle
from pathlib import Path

import torch
from scipy.io.wavfile import write

# If this does not work, make sure you added the SALT folder to your python path
# like described at the bottom of setup.sh
from SALT.anonymizer import Anonymizer

logging.basicConfig(level=logging.INFO)

assets_path = Path("/home/janneke/scripts/personal/ASR-stuff/SALT/assets")
input_wav_path = "/home/janneke/scripts/personal/ASR-stuff/data/emotions_dataset/Crema/1001_DFA_ANG_XX.wav"

if not os.path.exists("anonymizer.pkl"):
    logging.info("Loading anonymizer from github")
    anonymizer: Anonymizer = torch.hub.load("BakerBunker/SALT", "salt", trust_repo=True, pretrained=True, base=True, device="cuda")

    logging.info("Adding speakers to anonymizer")
    for file in assets_path.glob("*.pack"):
        print(file.stem)
        anonymizer.add_speaker(name=file.stem, preprocessed_file=file)

    # pickle anonymizer
    with open("anonymizer.pkl", "wb") as f:
        pickle.dump(anonymizer, f)

else:
    logging.info("Loading anonymizer from pickle")
    with open("anonymizer.pkl", "rb") as f:
        anonymizer = pickle.load(f)

# Create new wav
# run new speaker_dict for every new wav
speaker_dict = anonymizer.get_random_speaker()
wav = anonymizer.interpolate(
    input_wav_path,
    speaker_dict=speaker_dict,
    topk=4,  # K for k-NN
    chunksize=5,  # 5 sec for one chunk
    padding=0.5,  # pad 0.5 sec for head and tail each chunk
)


write(filename="test.wav", rate=16000, data=wav.cpu().numpy())
