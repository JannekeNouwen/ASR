import os
from pathlib import Path
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torchaudio

from utils import save_kaldi_format

LANGUAGE2TAG = {
    "dutch": "nl",
    "english": "en",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "polish": "pl",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
}


def read_enrolls(filepath):
    utts = []
    with open(filepath, "r") as f:
        for line in f:
            utts.append(line.strip())
    return utts


def read_trials(filepath):
    utts = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip().split()
            utts.append(line[1])
    return utts


def utt2spk_to_spk2utt(utt2spk):
    spk2utt = defaultdict(list)
    for utt, spk in utt2spk.items():
        spk2utt[spk].append(utt)
    return spk2utt


def get_audio_dur(audiopath):
    metadata = torchaudio.info(audiopath)
    return metadata.num_frames / metadata.sample_rate


def prepare_kaldi_format(utts, split, global_utt2spk, dataset_path, out_dir, data_type="mls"):
    wav_scp = dict()
    utt2dur = dict()

    deleted_utts = []
    for utt in utts:
        # if data_type == 'mls':
        #     mls_spk, mls_session, _ = utt.split('_')
        #     audio_path = dataset_path / split / 'audio' / mls_spk / mls_session / f'{utt}.flac'
        if data_type == "Tess":
            utt_split = utt.split("_")
            spk_id = global_utt2spk[utt]
            emotion = "_".join(utt_split[2:])
            audio_path = dataset_path / f"{spk_id}" / f"{utt}.wav"
            if not audio_path.exists():
                audio_path = dataset_path / f"{spk_id}_{emotion.capitalize()}" / f"{utt}.wav"
        else:
            raise NotImplementedError(f"Data type {data_type} not implemented")

        if not audio_path.exists():
            deleted_utts.append(utt)
            print(f"Audio does not exist: {audio_path}")
            print(emotion)
            continue
        wav_scp[utt] = str(audio_path)
        utt2dur[utt] = get_audio_dur(audio_path)

    utts = list(set(utts) - set(deleted_utts))

    # if data_type == 'mls':
    #     text = read_kaldi_format(dataset_path / split / 'transcripts.txt', values_as_string=True)
    #     text = {utt: sentence for utt, sentence in text.items() if utt in utts}
    if data_type == "Tess":
        # We do not have transcripts for this dataset
        text = {utt: "" for utt in utts}
    else:
        raise NotImplementedError(f"Data type {data_type} not implemented")
        # df = pd.read_csv(dataset_path / "validated.tsv", sep="\t")
        # df["path"] = df["path"].apply(lambda x: x.replace(".mp3", ""))
        # df = df.set_index("path")
        # text = {utt: df.loc[utt]["sentence"] for utt in utts}

    utt2spk = {utt: spk for utt, spk in global_utt2spk.items() if utt in utts}
    spk2utt = utt2spk_to_spk2utt(utt2spk)
    # if data_type == "mls":
    #     df = pd.read_csv(dataset_path / "metainfo.txt", delimiter="\s+\|\s+", engine="python")
    #     df = df.set_index("SPEAKER")
    #     spk2gender = df["GENDER"].to_dict()
    if data_type == "Tess":
        spk2gender = {spk: "F" for spk in spk2utt.keys()}
    else:
        raise NotImplementedError(f"Data type {data_type} not implemented")
        # spk2gender = {spk: spk[0] for spk in spk2utt.keys()}

    out_dir.mkdir(exist_ok=True, parents=True)
    save_kaldi_format(utt2spk, out_dir / "utt2spk")
    save_kaldi_format(spk2utt, out_dir / "spk2utt")
    # save_kaldi_format(text, out_dir / "text")
    save_kaldi_format(wav_scp, out_dir / "wav.scp")
    save_kaldi_format(utt2dur, out_dir / "utt2dur")
    save_kaldi_format(spk2gender, out_dir / "spk2gender")


def prepare_data(language, dataset_path, output_path, data_type="Tess"):
    utts, utt2spk = get_utts(data_type=data_type)
    # trials_data_path = Path(f'trials_data/{data_type}/{language}')
    # utt2spk_file = list(trials_data_path.glob('*_utt2spk'))[0]
    # utt2spk = read_kaldi_format(utt2spk_file)

    # for enrolls_file in trials_data_path.glob('*_enrolls'):
    # utts = read_enrolls(enrolls_file)
    # split = 'test' if 'test' in enrolls_file.name else 'dev'
    enroll_out_dir = output_path / f"{data_type}_enrolls"
    prepare_kaldi_format(
        utts=utts, global_utt2spk=utt2spk, dataset_path=dataset_path, out_dir=enroll_out_dir, data_type=data_type, split="test"
    )
    # write utts to file
    with open(enroll_out_dir / "enrolls", "w") as f:
        for utt in utts:
            f.write(f"{utt}\n")
    # copy(enrolls_file, enroll_out_dir / "enrolls")

    # for trials_file in trials_data_path.glob("*_trials*"):
    #     utts = read_trials(trials_file)
    #     split = "test" if "test" in trials_file.name else "dev"
    #     trials_out_dir = output_path / trials_file.name
    #     prepare_kaldi_format(
    #         utts=utts,
    #         split=split,
    #         global_utt2spk=utt2spk,
    #         dataset_path=dataset_path,
    #         out_dir=trials_out_dir,
    #         data_type=data_type,
    #     )
    #     copy(trials_file, trials_out_dir / "trials")


def get_utts(data_type="Tess"):
    """Utility function to get the utterances from the dataset.

    Utts seem to be some sort of identifier for a speech sample, so here I just use the file name.

    Args:
        data_type (str): The type of dataset. Currently only 'Tess' is supported.

    Returns:
        list: A list of utterances.
    """
    if data_type == "Tess":
        # List the folders in the dataset
        folders = [f for f in dataset_path.iterdir() if f.is_dir()]

        utts = []
        utt2spk = {}

        for folder in folders:
            path = Path(dataset_path / folder)
            # List the files in the folder
            for file in path.iterdir():
                last_folder = file.absolute().parent.name
                speaker = last_folder
                utts.append(f"{file.stem}")
                utt2spk[file.stem] = speaker


        return utts, utt2spk
    raise NotImplementedError(f"Data type {data_type} not implemented")


if __name__ == "__main__":
    dataset_path = Path("/home/janneke/scripts/personal/ASR-stuff/data/emotions_dataset/Tess")
    output_path = Path("/home/janneke/scripts/personal/ASR-stuff/speaker_anonymization/data")

    print(f"Prepare data for {dataset_path}")
    prepare_data(language="en", dataset_path=dataset_path, output_path=output_path, data_type="Tess")
