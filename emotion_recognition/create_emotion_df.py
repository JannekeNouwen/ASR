import glob

import pandas as pd


def main():
    audiofiles_dir = "./data/crema_d/audiofiles/*.wav"
    metadata = get_crema_metadata(audiofiles_dir)
    metadata.to_csv("./data/crema_d/metadata.csv")


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
                "emotion_id": unique_emotions[emotion],
            }
        )
    return pd.DataFrame(data)


def get_naomis_metadata(naomis_dir: str) -> pd.DataFrame:
    files = glob.glob(naomis_dir)
    # TODO(Naomi): Write a function to parse the accents dataset to get the metadata


if __name__ == "__main__":
    main()
