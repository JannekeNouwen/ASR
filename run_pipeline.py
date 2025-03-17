import json
from create_metadata import get_metadata
from wav2vec2_training import train
from inference import load_model, anonymize
import os
import wandb


def main():
    tasks = [
        #"age_recognition",
        "gender_recognition",
        #"emotion_recognition",
    ]

    for task in tasks:
        config_path = f"ASR/configs/{task}.json"
        run(config_path)


def run(config_path):

    with open(config_path, "r") as file:
        config = json.load(file)
    task = config["task"]
    dataset_name = config["dataset_name"]

    if not os.path.exists(f"data/{task}/{dataset_name}/audiofiles_anonymized"):
        model = load_model()
        anonymize(task, dataset_name, model)

    get_metadata(task=task, dataset_name=dataset_name)

    train(config, train_on="normal")
    wandb.finish()
    train(config, train_on="anonymized")
    wandb.finish()


if __name__ == "__main__":
    main()