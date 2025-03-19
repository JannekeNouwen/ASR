import json
import logging
import os
import random
from datetime import datetime

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset, load_from_disk
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Trainer, TrainingArguments

import wandb

# Set random state
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)


def train(config: dict, train_on: str) -> None:
    """Train a wav2vec model for classification or regression.

    Args:
        config: config for training the model.
        train_on: On which data to train. Choose one of normal/anonymized.
            (Evaluation happens on both, independent of which set is trained on)
    """
    dataset_name = config["dataset_name"]
    task = config["task"]
    use_wandb = config["use_wandb"]
    use_cached_dataset = config["use_cached_dataset"]
    match_metadata_on_filename = config["match_metadata_on_filename"]
    regression = config["regression"]
    label_column_name = config["label_column_name"]

    print(f"Running training for task {task} ({train_on}) with dataset {dataset_name}")

    metadata = pd.read_csv(f"./data/{task}/{dataset_name}/metadata.csv")
    if use_cached_dataset:
        logging.info("Loading cached dataset.")
        train_dataset = load_from_disk(f"./data/{task}/{dataset_name}/{dataset_name}_normal_train.hf")
        test_dataset = load_from_disk(f"./data/{task}/{dataset_name}/{dataset_name}_normal_test.hf")
        dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

        logging.info("Loading cached dataset.")
        anon_train_dataset = load_from_disk(f"./data/{task}/{dataset_name}/{dataset_name}_anonymized_train.hf")
        anon_test_dataset = load_from_disk(f"./data/{task}/{dataset_name}/{dataset_name}_anonymized_test.hf")
        anon_dataset = DatasetDict({"train": anon_train_dataset, "test": anon_test_dataset})
    else:
        logging.info("Loading dataset from audiofiles and metadata.")
        os.makedirs(f"./models/{task}", exist_ok=True)

        metadata, dataset, anon_dataset = load_data(
            f"./data/{task}/{dataset_name}/audiofiles",
            metadata=metadata,
            label_column_name=label_column_name,
            speaker_column_name="speaker",
            match_metadata_on_filename=match_metadata_on_filename
        )

        unique_speakers = metadata["speaker"].unique()

        logging.info(f"Making train/test split.")
        dataset, anon_dataset = make_train_test_split(
            dataset, anon_dataset, unique_speakers, dataset_name=dataset_name, task=task, split_ratio=0.8
        )
        print(f"Number of training examples: {len(dataset['train'])}")
        print(f"Number of testing examples: {len(dataset['test'])}")

    logging.info(f"Extracting features.")
    encoded_dataset, feature_extractor = preprocess_dataset(dataset, label_column_name)
    anon_encoded_dataset, _ = preprocess_dataset(anon_dataset, label_column_name)

    if regression:
        logging.info(f"Loading model for regression.")
        model = load_model(num_labels=1)
    else:
        num_labels = len(metadata["label_id"].unique())
        logging.info(f"Loading model with {num_labels} output classes.")
        model = load_model(num_labels=num_labels)

    date_str = datetime.today().strftime("%Y-%m-%d-%H.%M")

    logging.info(f"Loading trainer.")
    model_name = f"{task}-{dataset_name}-{train_on}-{date_str}"
    model_dir = f"./models/{task}/{model_name}"
    os.makedirs(model_dir)

    if train_on == "normal":
        trainer = get_trainer(
            model=model,
            encoded_dataset=encoded_dataset,
            feature_extractor=feature_extractor,
            use_wandb=use_wandb,
            model_dir=model_dir,
            run_name=model_name,
            compute_metrics=compute_metrics_regression if regression else compute_metrics_classification,
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
        )
    else:
        trainer = get_trainer(
            model=model,
            encoded_dataset=anon_encoded_dataset,
            feature_extractor=feature_extractor,
            use_wandb=use_wandb,
            model_dir=model_dir,
            run_name=model_name,
            compute_metrics=compute_metrics_regression if regression else compute_metrics_classification,
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
        )

    evaluate_model(trainer=trainer, encoded_dataset=encoded_dataset, model_name=model_name, task=task)
    evaluate_model(
        trainer=trainer, encoded_dataset=anon_encoded_dataset, model_name=model_name, task=task, anonymized=True
    )

    logging.info(f"Training model.")
    trainer.train()

    evaluate_model(trainer=trainer, encoded_dataset=encoded_dataset, model_name=model_name, task=task)
    evaluate_model(
        trainer=trainer, encoded_dataset=anon_encoded_dataset, model_name=model_name, task=task, anonymized=True
    )

    logging.info(f"Saving model")
    trainer.save_model(f"./models/{task}")


def evaluate_model(trainer: Trainer, encoded_dataset, model_name, task, anonymized=False):
    logging.info(f"Performing evaluation on train set.")
    output = trainer.evaluate(eval_dataset=encoded_dataset["train"], metric_key_prefix=f"train/{'anonymized' if anonymized else 'normal'}")
    logging.info(f"Evaluation on train set: {output}")
    with open(
        f"./models/{task}/{model_name}/train_set_metrics_{model_name}_on_train_set_{'anonymized' if anonymized else 'normal'}.json",
        "w",
    ) as file:
        json.dump(output, file, indent=4)

    logging.info(f"Performing evaluation on test set.")
    output = trainer.evaluate(eval_dataset=encoded_dataset["test"], metric_key_prefix=f"eval/{'anonymized' if anonymized else 'normal'}")
    logging.info(f"Evaluation on test set: {output}")
    with open(
        f"./models/{task}/{model_name}/test_set_metrics_{model_name}_on_test_set_{'anonymized' if anonymized else 'normal'}.json",
        "w",
    ) as file:
        json.dump(output, file, indent=4)


def load_data(
    audiofiles_dir: str, metadata: pd.DataFrame, label_column_name: str, speaker_column_name: str, match_metadata_on_filename: False
) -> tuple[pd.DataFrame, Dataset]:
    dataset = load_dataset(audiofiles_dir, name="default", split="train")
    anon_dataset = load_dataset(audiofiles_dir + "_anonymized", name="default", split="train")
    # if match_metadata_on_filename:
    #     dataset = dataset.add_column(label_column_name, metadata[label_column_name])
    #     anon_dataset = anon_dataset.add_column(label_column_name, [0]*len(anon_dataset))

    #     dataset = dataset.add_column(speaker_column_name, [0]*len(anon_dataset))
    #     anon_dataset = anon_dataset.add_column(speaker_column_name, [0]*len(anon_dataset))

        # for i in tqdm(range(len(dataset)), total=len(dataset)):
        #     dataset[i][label_column_name] = metadata.loc[metadata['audio_path'] == dataset[i]["audio"]["path"]][label_column_name]
        #     anon_dataset[i][label_column_name] = metadata.loc[metadata['audio_path'] == anon_dataset[i]["audio"]["path"]][label_column_name]

        #     dataset[i][speaker_column_name] = metadata.loc[metadata['audio_path'] == dataset[i]["audio"]["path"]][speaker_column_name]
        #     anon_dataset[i][speaker_column_name] = metadata.loc[metadata['audio_path'] == anon_dataset[i]["audio"]["path"]][speaker_column_name]
    # else:
    dataset = dataset.add_column(label_column_name, metadata[label_column_name])
    anon_dataset = anon_dataset.add_column(label_column_name, metadata[label_column_name])

    dataset = dataset.add_column(speaker_column_name, metadata[speaker_column_name])
    anon_dataset = anon_dataset.add_column(speaker_column_name, metadata[speaker_column_name])

    return metadata, dataset, anon_dataset


def make_train_test_split(
    dataset: Dataset,
    anon_dataset: Dataset,
    unique_speakers: np.ndarray,
    dataset_name: str,
    task: str,
    split_ratio: float = 0.8,
) -> DatasetDict:
    # Shuffle speakers and split them into train and test
    random.shuffle(unique_speakers)
    split_ratio = 0.8  # 80% speakers for train, 20% for test
    split_idx = int(len(unique_speakers) * split_ratio)

    train_speakers = set(unique_speakers[:split_idx])
    test_speakers = set(unique_speakers[split_idx:])

    # Apply filtering to create train and test sets
    train_set = dataset.filter(lambda example: example["speaker"] in train_speakers)
    anon_train_set = anon_dataset.filter(lambda example: example["speaker"] in train_speakers)
    test_set = dataset.filter(lambda example: example["speaker"] in test_speakers)
    anon_test_set = anon_dataset.filter(lambda example: example["speaker"] in test_speakers)

    logging.info(f"Saving dataset to disk.")
    train_set.save_to_disk(f"./data/{task}/{dataset_name}/{dataset_name}_normal_train.hf")
    anon_train_set.save_to_disk(f"./data/{task}/{dataset_name}/{dataset_name}_anonymized_train.hf")
    test_set.save_to_disk(f"./data/{task}/{dataset_name}/{dataset_name}_normal_test.hf")
    anon_test_set.save_to_disk(f"./data/{task}/{dataset_name}/{dataset_name}_anonymized_test.hf")

    dataset = DatasetDict({"train": train_set, "test": test_set})
    anon_dataset = DatasetDict({"train": anon_train_set, "test": anon_test_set})

    # Verify no overlap
    assert (
        len(set(dataset["train"]["speaker"]).intersection(set(dataset["test"]["speaker"]))) == 0
    ), "Speaker overlap detected between train and test!"
    assert (
        len(set(dataset["train"]["speaker"]).intersection(set(anon_dataset["test"]["speaker"]))) == 0
    ), "Speaker overlap detected between train (normal) and test (anon)!"

    return dataset, anon_dataset


def preprocess_dataset(dataset: DatasetDict, label_column_name) -> tuple[DatasetDict, AutoFeatureExtractor]:
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000),)
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    def preprocess_function(examples: dict) -> dict:
        audio_arrays = [x["array"] for x in examples["audio"]]
        return feature_extractor(
            audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
        )

    encoded_dataset = dataset.map(preprocess_function, remove_columns="audio", batched=True)
    encoded_dataset = encoded_dataset.rename_column(label_column_name, "label")
    return encoded_dataset, feature_extractor


def load_model(num_labels: int) -> AutoModelForAudioClassification:
    return AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", num_labels=num_labels, ignore_mismatched_sizes=True)


def compute_metrics_classification(eval_preds):
    metric_accuracy = evaluate.load("accuracy")
    metric_precision = evaluate.load("precision")
    metric_recall = evaluate.load("recall")
    metric_f1 = evaluate.load("f1")

    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Compute metrics with macro averaging
    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
    precision = metric_precision.compute(predictions=predictions, references=labels, average="macro")
    recall = metric_recall.compute(predictions=predictions, references=labels, average="macro")
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }

def compute_metrics_regression(eval_preds):
    metric_mse = evaluate.load("mse")
    metric_mae = evaluate.load("mae")
    metric_r2 = evaluate.load("r_squared")
    
    predictions, labels = eval_preds  # No argmax, as outputs are continuous
    
    # Compute regression metrics
    mse = metric_mse.compute(predictions=predictions, references=labels)
    mae = metric_mae.compute(predictions=predictions, references=labels)
    r2 = metric_r2.compute(predictions=predictions, references=labels)

    return {
        "mse": mse["mse"],
        "mae": mae["mae"],
        "r2": r2["r_squared"],
    }


def get_trainer(
    model: AutoModelForAudioClassification,
    encoded_dataset: DatasetDict,
    feature_extractor: AutoFeatureExtractor,
    use_wandb: bool,
    model_dir: str,
    run_name: str,
    compute_metrics,
    batch_size: int = 16,
    epochs: int = 10,
    learning_rate: float = 3e-5
) -> Trainer:
    if use_wandb:
        wandb.login()
        os.environ["WANDB_PROJECT"] = "ASR2025"  # name your W&B project

    training_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=8,
        report_to="wandb" if use_wandb else "none",
        run_name=run_name,
        logging_strategy="steps",
        logging_steps=int(len(encoded_dataset["train"]) * epochs / batch_size / 20),
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )


if __name__ == "__main__":
    train()
