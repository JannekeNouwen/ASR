import logging
import os
import pickle
from pathlib import Path
import glob
import torch
from scipy.io.wavfile import write
from tqdm import tqdm

# If this does not work, make sure you added the SALT folder to your python path
# like described at the bottom of setup.sh
from SALT.anonymizer import Anonymizer

logging.basicConfig(level=logging.INFO)


def main():
    task_name = "emotion_recognition"
    dataset_name = "crema_d"

    model = load_model()
    anomymize(task_name, dataset_name, model)

def load_model():
    assets_path = Path("./SALT/assets")

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
    
    return anonymizer


def anomymize(task_name, dataset_name, model):
    output_dir = f"./data/{task_name}/{dataset_name}/audiofiles_anonymized"
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(f"./data/{task_name}/{dataset_name}/audiofiles/*.wav")
    for audiofile in tqdm(files, total=len(files)):
        speaker_dict = model.get_random_speaker()
        wav = model.interpolate(
            audiofile,
            speaker_dict=speaker_dict,
            topk=4,  # K for k-NN
            chunksize=5,  # 5 sec for one chunk
            padding=0.5,  # pad 0.5 sec for head and tail each chunk
        )

        new_audiofile_path = f"{output_dir}/{Path(audiofile).stem}.wav"
        write(filename=new_audiofile_path, rate=16000, data=wav.cpu().numpy())

if __name__ == "__main__":
    main()