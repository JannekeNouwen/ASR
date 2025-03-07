from tqdm.contrib.concurrent import process_map
import time
import logging
from torch.multiprocessing import set_start_method
from itertools import  repeat
import torch
import soundfile
from pathlib import Path
import resampy

from .text import Text
from .recognition import ImsASR, WhisperASR, Wav2Vec2ASR, MMSASR
from utils import read_kaldi_format, setup_logger

# from speaker_anonymization.anonymization.modules.text.speech_recognition import ASRDataset, run_process
from speaker_anonymization.anonymization.modules.text.speech_recognition import SpeechRecognition

set_start_method('spawn', force=True)
logger = setup_logger(__name__)


class ASRDatasetEmotions(torch.utils.data.Dataset):
    def __init__(self, utt2spk, wav_scp, utt2dur, already_recognized_utts, utterance_list, eval=False):
        self.utterances = []
        for utt, spk in utt2spk.items():
            if utt not in wav_scp:
                continue
            if eval:
                if float(utt2dur[utt]) > 30.0:  # whisper has problems with utterances longer than 30 seconds, there are a few in LibriSpeech
                    continue
            if utt in already_recognized_utts:
                continue
            if utterance_list and utt not in utterance_list:
                continue
            if utt in wav_scp:
                self.utterances.append((utt, spk, wav_scp[utt]))

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt, spk, wav_path = self.utterances[idx]
        speech, rate = soundfile.read(wav_path)
        if rate != 16000:
            speech = resampy.resample(speech, rate, 16000)
        return {'raw': speech, 'sampling_rate': 16000, 'utt': utt, 'spk': spk}


def run_process(params):
    utt2spk, wav_scp, utt2dur, already_recognized_utts, utterance_list, asr_model, out_dir, eval, save_intermediate, \
        sleep, job_id = params
    time.sleep(sleep)
    asr_dataset = ASRDatasetEmotions(utt2spk=utt2spk, wav_scp=wav_scp, utt2dur=utt2dur,
                             already_recognized_utts=already_recognized_utts, utterance_list=utterance_list, eval=eval)
    return asr_model.recognize_speech_of_dataset(asr_dataset, out_dir=out_dir, save_intermediate=save_intermediate,
                                                 job_id=job_id)



class SpeechRecognitionCustom(SpeechRecognition):

    def recognize_speech(self, dataset_path, dataset_name=None, utterance_list=None):
        dataset_name = dataset_name if dataset_name else dataset_path.name
        dataset_results_dir = self.results_dir / dataset_name if self.save_intermediate else Path('')

        if self.asr_models is None:
            return self._load_gold_transcripts(dataset_path)

        utt2spk = read_kaldi_format(dataset_path / 'utt2spk')
        texts = Text(is_phones=self.is_phones)

        if (dataset_results_dir / 'text').exists() and not self.force_compute:
            # if the text created from this ASR model already exists for this dataset and a computation is not
            # forced, simply load the text
            texts.load_text(in_dir=dataset_results_dir)

        if len(texts) > 0:
            logger.info(f'No speech recognition necessary for {len(texts)} of {len(utt2spk)} utterances')
        # otherwise, recognize the speech
        dataset_results_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f'Recognize speech of {len(utt2spk) - len(texts)} utterances...')
        wav_scp = read_kaldi_format(dataset_path / 'wav.scp')
        utt2dur = read_kaldi_format(dataset_path / 'utt2dur')

        save_intermediate = self.save_intermediate and not utterance_list
        start = time.time()
        if self.n_processes == 1:
            params = [utt2spk, wav_scp, utt2dur, texts.utterances, utterance_list, self.asr_models[0],
                        dataset_results_dir, self.eval, save_intermediate, 0, None]
            new_texts = [run_process(params)]
        else:
            sleeps = [10 * i for i in range(self.n_processes)]
            utt2spk_jobs = [{k: v for k, v in list(utt2spk.items())[i::self.n_processes]}
                            for i in range(self.n_processes)]
            params = zip(utt2spk_jobs, # utterances to recognize
                            repeat(wav_scp), # wav paths
                            repeat(utt2dur), # durations
                            repeat(texts.utterances), # already recognized utterances
                            repeat(utterance_list), # sub list of utterances
                            self.asr_models, # asr model
                            repeat(dataset_results_dir),  # out_dir
                            repeat(self.eval),
                            repeat(save_intermediate),  # whether to save intermediate results
                            sleeps, # avoid starting all processes at same time
                            list(range(self.n_processes))) # job_id
            new_texts = process_map(run_process, params, max_workers=self.n_processes)


            end = time.time()
            total_time = round(end - start, 2)
            logger.info(f'Total time for speech recognition: {total_time} seconds ({round(total_time / 60, 2)} minutes / '
                  f'{round(total_time / 60 / 60, 2)} hours)')

            texts = self._combine_texts(main_text_instance=texts, additional_text_instances=new_texts)
            if save_intermediate:
                texts.save_text(out_dir=dataset_results_dir)
                self._remove_temp_files(out_dir=dataset_results_dir)

        return texts
