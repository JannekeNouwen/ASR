import os
import random

import evaluate
import numpy as np
import pandas as pd
from datasets import Audio, Dataset, DatasetDict, load_dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Trainer, TrainingArguments

import wandb

# Set random state
random.seed(42)


def main():
    # First run create_emotion_df.py to create the metadata.csv file

    # I hope that you only need to change these 3 variables to run this script:
    dataset_name = "crema_d"
    task = "emotion_recognition"
    use_wandb = True

    os.makedirs(f"./models/{task}", exist_ok=True)

    metadata, dataset = load_data(
        f"./data/{dataset_name}/audiofiles",
        metadata_csv_path=f"./data/{dataset_name}/metadata.csv",
        label_column_name="emotion_id",
        speaker_column_name="speaker",
    )

    unique_speakers = metadata["speaker"].unique()

    dataset = make_train_test_split(dataset, unique_speakers, split_ratio=0.8)
    print(f"Number of training examples: {len(dataset['train'])}")
    print(f"Number of testing examples: {len(dataset['test'])}")

    dataset.save_to_disk(f"./data/{dataset_name}/{dataset_name}.hf")

    encoded_dataset, feature_extractor = preprocess_dataset(dataset)

    model = load_model(num_labels=len(metadata["emotion_id"].unique()))

    trainer = get_trainer(model, encoded_dataset, feature_extractor, use_wandb=use_wandb)

    output = trainer.evaluate(eval_dataset=encoded_dataset["test"])
    print(output)

    trainer.train()

    output = trainer.evaluate(eval_dataset=encoded_dataset["test"])
    print(output)

    trainer.save_model(f"./models/{task}")


def load_data(
    audiofiles_dir: str, metadata_csv_path: str, label_column_name: str, speaker_column_name: str
) -> tuple[pd.DataFrame, Dataset]:
    dataset = load_dataset(audiofiles_dir, name="default", split="train")
    metadata = pd.read_csv(metadata_csv_path)
    dataset = dataset.add_column(label_column_name, metadata[label_column_name])
    dataset = dataset.add_column(speaker_column_name, metadata[speaker_column_name])
    print(f"Example: {dataset[0]}")
    return metadata, dataset


def make_train_test_split(dataset: Dataset, unique_speakers: np.ndarray, split_ratio: float = 0.8) -> DatasetDict:
    # Shuffle speakers and split them into train and test
    random.shuffle(unique_speakers)
    split_ratio = 0.8  # 80% speakers for train, 20% for test
    split_idx = int(len(unique_speakers) * split_ratio)

    train_speakers = set(unique_speakers[:split_idx])
    test_speakers = set(unique_speakers[split_idx:])

    # Apply filtering to create train and test sets
    train_set = dataset.filter(lambda example: example["speaker"] in train_speakers)
    test_set = dataset.filter(lambda example: example["speaker"] in test_speakers)

    # Verify no overlap
    dataset = DatasetDict({"train": train_set, "test": test_set})

    assert len(set(dataset["train"]["speaker"]).intersection(set(dataset["test"]["speaker"]))) == 0, (
        "Speaker overlap detected between train and test!"
    )

    # Combine into DatasetDict
    return dataset


def preprocess_dataset(dataset: DatasetDict) -> tuple[DatasetDict, AutoFeatureExtractor]:
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    def preprocess_function(examples: dict) -> dict:
        audio_arrays = [x["array"] for x in examples["audio"]]
        return feature_extractor(
            audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
        )

    encoded_dataset = dataset.map(preprocess_function, remove_columns="audio", batched=True)
    encoded_dataset = encoded_dataset.rename_column("emotion_id", "label")
    return encoded_dataset, feature_extractor


def load_model(num_labels: int) -> AutoModelForAudioClassification:
    return AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", num_labels=num_labels)


def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def get_trainer(
    model: AutoModelForAudioClassification,
    encoded_dataset: DatasetDict,
    feature_extractor: AutoFeatureExtractor,
    use_wandb: bool,
) -> Trainer:
    if use_wandb:
        wandb.login()
        os.environ["WANDB_PROJECT"] = "ASR2025"  # name your W&B project
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    batch_size = 16
    epochs = 5
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=8,
        report_to="wandb" if use_wandb else "none",
        logging_strategy="steps",
        logging_steps=5890 * epochs / batch_size / 20,
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
    main()
