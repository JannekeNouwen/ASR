import glob
import pandas as pd


def main():
    crema_dir = "./data/crema_d/audiofiles/*.wav"
    emotion_df = get_crema_data(crema_dir)
    emotion_df.to_csv("./data/crema_d/crema.csv")

    # iemocap_dir = "./data/iemocap/audiofiles/*.wav"


def get_crema_data(crema_dir: str) -> pd.DataFrame:
    files = glob.glob(crema_dir)
    data = []
    unique_emotions = {}
    for file in files:
        actor, sentence, emotion, emotion_level = file.split("/")[-1].split(".")[0].split("_")

        if emotion not in unique_emotions:
            unique_emotions[emotion] = len(unique_emotions)

        data.append(
            {"speaker": actor, "emotion": emotion.lower(), "emotion_level": emotion_level, "sentence": sentence, "emotion_id": unique_emotions[emotion]}
        )
    return pd.DataFrame(data)


if __name__ == "__main__":
    main()
