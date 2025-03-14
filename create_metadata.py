import glob

import pandas as pd
import json


def main():
    config_path = "ASR/configs/emotion_recognition.json"
    with open(config_path, "r") as file:
        config = json.load(file)
    get_metadata(config)


def get_metadata(task: str, dataset_name: str):
    audiofiles_dir = f"./data/{task}/{dataset_name}/audiofiles/*.wav"
    metadata = DATASET_TO_FUNC[dataset_name](audiofiles_dir)
    metadata.to_csv(f"./data/{task}/{dataset_name}/metadata.csv")


def get_crema_metadata(crema_dir: str) -> pd.DataFrame:
    files = glob.glob(crema_dir)
    data = []
    unique_emotions = {}
    for file in files:
        actor, sentence, emotion, emotion_level = file.split("/")[-1].split(".")[0].split("_")

        if emotion not in unique_emotions:
            unique_emotions[emotion] = len(unique_emotions)

        data.append(
            {
                "speaker": actor,
                "emotion": emotion.lower(),
                "emotion_level": emotion_level,
                "sentence": sentence,
                "label_id": unique_emotions[emotion],
            }
        )
    return pd.DataFrame(data)


def get_naomis_metadata(naomis_dir: str) -> pd.DataFrame:
    files = glob.glob(naomis_dir)
    # TODO(Naomi): Write a function to parse the accents dataset to get the metadata


DATASET_TO_FUNC = {"crema_d": get_crema_metadata, "naomi": get_naomis_metadata}

if __name__ == "__main__":
    main()
