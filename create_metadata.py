import glob

import pandas as pd
import json
import os


# def main():
#     config_path = "ASR/configs/accent_recognition.json"
#     with open(config_path, "r") as file:
#         config = json.load(file)
#     task = config["task"]
#     dataset_name = config["dataset_name"]
#     label_column_name = config["label_column_name"]
#     get_metadata(task=task, dataset_name=dataset_name,label_column_name=label_column_name,undersampling=True)


def get_metadata(task: str, dataset_name: str, label_column_name: str, undersampling = False):
    audiofiles_dir = f"./data/{task}/{dataset_name}/audiofiles/*.wav"
    metadata = DATASET_TO_FUNC[dataset_name](audiofiles_dir)

    if undersampling and len(set(metadata[label_column_name].value_counts())) > 1:
        lowest_value = metadata[label_column_name].value_counts().idxmin()
        smallest_subset = metadata.loc[metadata[label_column_name] == lowest_value, :]
        number_of_samples = len(smallest_subset)

        subset_list = []
        for label in metadata[label_column_name].unique():
            subset = metadata.loc[metadata[label_column_name] == label, :]
            subset_list.append(subset.sample(n=number_of_samples, replace=False))
        metadata = pd.concat(subset_list)

        # Remove files not used in sampling
        files = glob.glob(os.path.abspath(audiofiles_dir))
        pwd = files[0].split('/data/')[0]
        list_audiopaths = list(metadata['audio_path'].apply(lambda x: ''.join([pwd,x[1:]])))
        for file in files:
            if not file in list_audiopaths:
                os.remove(file)

        anon_files = glob.glob(os.path.abspath(audiofiles_dir.replace("audiofiles", "audiofiles_anonymized")))
        for file in anon_files:
            if not file in list([i.replace("audiofiles", "audiofiles_anonymized") for i in list_audiopaths]):
                os.remove(file)

    metadata.to_csv(f"./data/{task}/{dataset_name}/metadata.csv")


def get_crema_metadata(crema_dir: str, annotation_file: str = None) -> pd.DataFrame:
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
                "audio_path": file
            }
        )
    return pd.DataFrame(data)


def get_accent_metadata(accent_dir: str) -> pd.DataFrame:
    files = glob.glob(accent_dir)
    unique_accents = {}
    data = []
    for file in files:
        speaker, accent,_ = file.split("/")[-1].split('_')
        if accent not in unique_accents:
            unique_accents[accent] = len(unique_accents)
        data.append({"speaker":speaker,'accent':accent, 'label_id':unique_accents[accent],'audio_path':file})

    metadata = pd.DataFrame(data)
    return metadata.sort_values("speaker")


def get_bvc_one_sentence_metadata(bvc_one_sentence_dir: str, annotation_file: str = f"ASR/configs/bvc_annotation.csv") -> pd.DataFrame:
    files = glob.glob(os.path.abspath(bvc_one_sentence_dir))
    id_to_filename = {int(file.split("/")[-1].split("_")[2]): file for file in files if "VE" in file.split("/")[-1]}
    data = []
    unique_genders = {}

    annotation_file = pd.read_csv(annotation_file)
    for i, row in annotation_file.iterrows():
        gender = row["Sex"]
        audio_id = row["New_ID"]

        if gender not in unique_genders:
            unique_genders[gender] = len(unique_genders)

        if not audio_id in id_to_filename:
            continue
        audio_path = id_to_filename[int(audio_id)]

        if audio_path:
            data.append(
                {
                    "speaker": audio_id,
                    "gender": gender,
                    "label_id": unique_genders[gender],
                    "audio_path": audio_path
                }
            )
        else:
            print(f"Audio path {audio_path} not found.")
    metadata = pd.DataFrame(data)

    files = glob.glob(os.path.abspath(bvc_one_sentence_dir))
    for file in files:
        if not file in list(metadata["audio_path"]):
            os.remove(file)

    anon_files = glob.glob(os.path.abspath(bvc_one_sentence_dir.replace("audiofiles", "audiofiles_anonymized")))
    for file in anon_files:
        if not file in list([i.replace("audiofiles", "audiofiles_anonymized") for i in metadata["audio_path"]]):
            os.remove(file)

    return metadata.sort_values("speaker")


def get_bvc_multiple_sentences_metadata(bvc_multiple_sentences_dir: str, annotation_file: str = f"ASR/configs/bvc_annotation.csv") -> pd.DataFrame:
    files = glob.glob(os.path.abspath(bvc_multiple_sentences_dir))
    id_to_filename = {int(file.split("/")[-1].split("_")[2]): file for file in files if "VE" in file.split("/")[-1]}
    data = []
    unique_genders = {}

    annotation_file = pd.read_csv(annotation_file)
    for i, row in annotation_file.iterrows():
        gender = row["Sex"]
        age = row["Age"]
        audio_id = row["New_ID"]

        if gender not in unique_genders:
            unique_genders[gender] = len(unique_genders)

        if not audio_id in id_to_filename:
            continue
        audio_path = id_to_filename[int(audio_id)]
        sentence_index = int(id_to_filename[int(audio_id)].split("_")[-1][2])

        for sentence_index in range(1, 6):
            if os.path.exists(f"{audio_path.split('VE')[0]}VE{sentence_index}.wav"):
                data.append(
                    {
                        "speaker": audio_id,
                        "gender": gender,
                        "sentence_index": sentence_index,
                        "label_id": unique_genders[gender],
                        "age": int(age),
                        "audio_path": f"{audio_path.split('VE')[0]}VE{sentence_index}.wav"
                    }
                )
    
    metadata = pd.DataFrame(data)

    files = glob.glob(os.path.abspath(bvc_multiple_sentences_dir))
    for file in files:
        if not file in list(metadata["audio_path"]):
            os.remove(file)

    anon_files = glob.glob(os.path.abspath(bvc_multiple_sentences_dir.replace("audiofiles", "audiofiles_anonymized")))
    for file in anon_files:
        if not file in list([i.replace("audiofiles", "audiofiles_anonymized") for i in metadata["audio_path"]]):
            os.remove(file)

    return metadata.sort_values("speaker")

DATASET_TO_FUNC = {"crema_d": get_crema_metadata, "speech_accent_archive": get_accent_metadata, "bvc_one_sentence": get_bvc_one_sentence_metadata, "bvc_multiple_sentences": get_bvc_multiple_sentences_metadata}

# if __name__ == "__main__":
#     main()
