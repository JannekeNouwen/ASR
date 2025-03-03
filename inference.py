from argparse import ArgumentParser
from pathlib import Path
import subprocess
import torch
import os
import shutil
from huggingface_hub import hf_hub_download

import speaker_anonymization
from speaker_anonymization.inference import InferenceAnonymizer, InferenceASR, InferenceTTS
# from evaluation import prepare_evaluation_data, copy_evaluation_results


def create_clean_dir(dir_name:Path):
    if dir_name.exists():
        remove_contents_in_dir(dir_name)
    else:
        dir_name.mkdir(exist_ok=True, parents=True)


def remove_contents_in_dir(dir_name: Path):
    # solution from https://stackoverflow.com/a/56151260
    for path in dir_name.glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def create_result_folders(settings, force_compute, results_dir):
    model_result_dirs = {}

    khz = f'{settings["output_sr"] // 1000}kHz'
    anon_model_name = settings['anonymizer']
    asr_model_name = settings['asr']
    tts_model_name = f'{settings["tts_hifigan"]}_{settings["tts_fastspeech"]}_{khz}'

    # anon
    anon_results_dir = results_dir / 'speaker_embeddings' / anon_model_name
    if not anon_results_dir.exists() or 'anon' in force_compute:
        create_clean_dir(anon_results_dir)
    model_result_dirs['anon'] = anon_results_dir

    # asr
    asr_results_dir = results_dir / 'transcription' / asr_model_name
    if not asr_results_dir.exists() or 'asr' in force_compute:
        create_clean_dir(asr_results_dir)
    model_result_dirs['asr'] = asr_results_dir

    # tts
    tts_results_dir = results_dir / 'speech' / tts_model_name / anon_model_name / asr_model_name
    if not tts_results_dir.exists() or 'tts' in force_compute:
        create_clean_dir(tts_results_dir)
    model_result_dirs['tts'] = tts_results_dir

    return model_result_dirs


def run_pipeline(device, settings, datasets, dirs):
    print('Set up components...')
    anonymizer = InferenceAnonymizer(settings['anonymizer'], data_dir=dirs['data'], model_dir=dirs["models"],
                                     results_dir=dirs['results']['anon'], vectors_dir=dirs['vectors'], device=device)
    asr = InferenceASR(settings['asr'], device=device, data_dir=dirs['data'], model_dir=dirs['models'],
                       results_dir=dirs['results']['asr'])
    tts = InferenceTTS(hifigan_model_name=settings['tts_hifigan'], fastspeech_model_name=settings['tts_fastspeech'], embed_model=settings["tts_embedding_model"],
                       model_dir=dirs['models'], results_dir=dirs['results']['tts'], device=device,
                       data_dir=dirs['data'], output_sr=settings['output_sr'])

    anon_wav_scps = {}

    for i, dataset in enumerate(datasets):
        print(f'{i + 1}/{len(datasets)}: Processing {dataset}...')
        anon_embeddings, new_anon = anonymizer.anonymize_embeddings(dataset=dataset)
        texts, utt2spk, new_text = asr.recognize_speech(dataset=dataset)
        wav_scp, _ = tts.read_texts(dataset=dataset, texts=texts, anon_embeddings=anon_embeddings, utt2spk=utt2spk,
                                    force_compute=(new_anon or new_text),
                                    text_is_phonemes=settings['text_is_phonemes'])
        anon_wav_scps[dataset] = wav_scp
        print('Done')
    return anon_wav_scps


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    args = parser.parse_args()

    # Settings
    gpu = args.gpu     # None for CPU, integer for GPU ID
    settings = {
        'datasets': ['libri_dev', 'libri_test', 'vctk_dev', 'vctk_test'],
        'anonymizer': 'gan',  # name of anonymization model
        'asr': 'asr_improved_tts-phn_en.zip',  # name of ASR model
        'tts_hifigan': 'best.pt',  # name of TTS HiFiGAN model
        'tts_fastspeech': 'trained_on_ground_truth_phonemes.pt',  # name of TTS FastSpeech2 model
        'tts_embedding_model': 'embedding_gan.pt',
        'output_sr': 48000
    }
    force_compute = []  # options: 'anon', 'asr', 'tts'
    settings['text_is_phonemes'] = '-phn' in settings['asr']


    # Some static variables
    data_dir = Path('Voice-Privacy-Challenge-2020', 'baseline', 'data')
    vectors_dir = Path('original_speaker_embeddings')
    models_dir = Path(os.path.dirname(speaker_anonymization.__file__) + "/models")
    # models_dir = Path('models')
    results_dir = Path('results')
    
    embedding_model_dir = models_dir / 'tts' / 'ToucanTTS'
    os.makedirs(embedding_model_dir, exist_ok=True)
    # If directory is empty, download the model
    if not os.listdir(embedding_model_dir):
        embedding_gan_path = hf_hub_download(cache_dir=embedding_model_dir, repo_id="Flux9665/ToucanTTS", filename="embedding_gan.pt")
        # move the downloaded model to the correct directory
        shutil.move(embedding_gan_path, embedding_model_dir / "embedding_model_tts.pt")


    model_result_dirs = create_result_folders(settings=settings, force_compute=force_compute, results_dir=results_dir)

    dirs = {
        'data': data_dir,
        'models': models_dir,
        'results': model_result_dirs,
        'vectors': vectors_dir
    }

    # the challenge's eval scripts require the data to be at a specific location
    eval_data_dir = Path('Voice-Privacy-Challenge-2020', 'baseline', 'data')

    if not torch.cuda.is_available():
        gpu = None
    device = torch.device(f'cuda:{gpu}') if gpu is not None else torch.device('cpu')

    dataset_splits = {
        'libri': ['trials_f', 'trials_m', 'enrolls'],
        'vctk': ['trials_f_all', 'trials_m_all', 'enrolls']
    }

    datasets = [f'{dset}_{split}' for dset in settings['datasets'] for split in dataset_splits[dset.split('_')[0]]]

    anon_wav_scps = run_pipeline(device=device, settings=settings, datasets=datasets, dirs=dirs)

    # Evaluation
    # print(f'Prepare evaluation data for {datasets}...')
    # prepare_evaluation_data(dataset_list=datasets, anon_wav_scps=anon_wav_scps, orig_data_path=data_dir,
    #                         anon_vectors_path=model_result_dirs['anon'], output_path=eval_data_dir)

    # if 'vctk_dev' in settings['datasets']:
    #     print('Make anon subsets for vctk_dev...')
    #     subprocess.run(['./evaluation/run_make_vctk_anon_subsets.sh', '--split', 'dev'], check=True)
    # if 'vctk_test' in settings['datasets']:
    #     print('Make anon subsets for vctk_test...')
    #     subprocess.run(['./evaluation/run_make_vctk_anon_subsets.sh', '--split', 'test'], check=True)

    # print('Perform evaluation...')
    # subprocess.run(['./evaluation/run_evaluation.sh', '--mcadams', 'false', '--gpu', str(gpu) or 'cpu',
    #                 *settings['datasets']], check=True)

    # # Copy the evaluation results to our results directory
    # copy_evaluation_results(results_dir=results_dir / 'evaluation', eval_dir=eval_data_dir, settings=settings)

    # subprocess.run(['./utils/run_cleanup.sh'], check=True)

