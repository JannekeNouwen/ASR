import json
from create_metadata import get_metadata
from wav2vec2_training import train, prepare_data
from inference import load_model, anonymize
import os
import wandb
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    tasks = [
        "age_recognition",
        "gender_recognition",
        "emotion_recognition",
        "accent_recognition"
    ]

    num_repetitions = 10

    for task in tasks:
        config_path = f"ASR/configs/{task}.json"
        config, metadata = prepare_run(config_path)
        output = []
        for repetition in range(num_repetitions):
            metrics_dict = run(config, repetition, metadata=metadata)
            output.append(metrics_dict)

        analyse_output(output, output_dir=f"./models/{task}/{task}_")
        

def analyse_output(output, output_dir: str):
    summary_dict = {
        "model_normal": { 
            "data_normal": {
                "train": {
                    "loss": [],
                    "f1": [],
                    "accuracy": [],
                    "precision": [],
                    "recall": [],
                },
                "test": {
                    "loss": [],
                    "f1": [],
                    "accuracy": [],
                    "precision": [],
                    "recall": [],
                }
            },
            "data_anon": {
                "train": {
                    "loss": [],
                    "f1": [],
                    "accuracy": [],
                    "precision": [],
                    "recall": [],
                },
                "test": {
                    "loss": [],
                    "f1": [],
                    "accuracy": [],
                    "precision": [],
                    "recall": [],
                }
            }
        },
        "model_anon":{
            "data_normal": {
                "train": {
                    "loss": [],
                    "f1": [],
                    "accuracy": [],
                    "precision": [],
                    "recall": [],
                },
                "test": {
                    "loss": [],
                    "f1": [],
                    "accuracy": [],
                    "precision": [],
                    "recall": [],
                }
            },
            "data_anon": {
                "train": {
                    "loss": [],
                    "f1": [],
                    "accuracy": [],
                    "precision": [],
                    "recall": [],
                },
                "test": {
                    "loss": [],
                    "f1": [],
                    "accuracy": [],
                    "precision": [],
                    "recall": [],
                }
            }
        }
    }

    for repetiton, metrics_dict in enumerate(output):
        for model in ["normal", "anon"]:  
            for data in ["normal", "anon"]:
                for split in ["train", "test"]:
                    for metric in ["loss", "f1", "accuracy", "precision", "recall"]:
                        summary_dict[f"model_{model}"][f"data_{data}"][split][metric].append(metrics_dict[f"model_{model}"][f"data_{data}"][f"split_{split}"][f"{split if split == 'train' else 'eval'}/{'normal' if data == 'normal' else 'anonymized'}_{metric}"])
 
    with open(f"{output_dir}repetitions_metrics.json", "w") as file:
        json.dump(summary_dict, indent=4, fp=file)

# def analyse_output_2(output_dir, summary_dict):
    for split in ["train", "test"]:
        with open(f"{output_dir}repetitons_results_{split}.txt", "w") as file:
            file.write("Compare metrics from model trained on normal data evaluated on normal and anonymized data\n")
            for metric in ["loss", "f1", "accuracy", "precision", "recall"]:
                x = summary_dict[f"model_normal"]["data_normal"][split][metric]
                y = summary_dict[f"model_normal"]["data_anon"][split][metric]
                res = wilcoxon(x, y)
                # file.write(f"\t{metric.capitalize()} normal data mean: {np.mean(normal_normal_train_metric)}\n")
                # file.write(f"\t{metric.capitalize()} normal data std: {np.std(normal_normal_train_metric)}\n")
                # file.write(f"\t{metric.capitalize()} anon data mean: {np.mean(normal_anon_train_metric)}\n")
                # file.write(f"\t{metric.capitalize()} anon data std: {np.std(normal_anon_train_metric)}\n")
                file.write(f"\t{metric.capitalize()} wilcoxon statistic: {res.statistic}\n")
                file.write(f"\t{metric.capitalize()} wilcoxon p-value: {res.pvalue}\n")
                file.write("\n")

            file.write("Compare metrics from model trained on normal data to model trained on anonymized data. Both evaluated on normal data\n")
            for metric in ["loss", "f1", "accuracy", "precision", "recall"]:
                x = summary_dict[f"model_normal"]["data_normal"][split][metric]
                y = summary_dict[f"model_anon"]["data_normal"][split][metric]
                res = wilcoxon(x, y)
                file.write(f"\t{metric.capitalize()} wilcoxon statistic: {res.statistic}\n")
                file.write(f"\t{metric.capitalize()} wilcoxon p-value: {res.pvalue}\n")
                file.write("\n")

            file.write("Compare metrics from model trained on normal data to model trained on anonymized data. Both evaluated on anonymized data\n")
            for metric in ["loss", "f1", "accuracy", "precision", "recall"]:
                x = summary_dict[f"model_normal"]["data_anon"][split][metric]
                y = summary_dict[f"model_anon"]["data_anon"][split][metric]
                res = wilcoxon(x, y)
                file.write(f"\t{metric.capitalize()} wilcoxon statistic: {res.statistic}\n")
                file.write(f"\t{metric.capitalize()} wilcoxon p-value: {res.pvalue}\n")
                file.write("\n")

            file.write("Compare metrics from model trained on normal data to model trained on anonymized data. Normal model evaluated on normal data, anon model on anon data\n")
            for metric in ["loss", "f1", "accuracy", "precision", "recall"]:
                x = summary_dict[f"model_normal"]["data_normal"][split][metric]
                y = summary_dict[f"model_anon"]["data_anon"][split][metric]
                res = wilcoxon(x, y)
                file.write(f"\t{metric.capitalize()} wilcoxon statistic: {res.statistic}\n")
                file.write(f"\t{metric.capitalize()} wilcoxon p-value: {res.pvalue}\n")
                file.write("\n")


    for metric in ["loss", "f1", "accuracy", "precision", "recall"]:
        # Performance on anonymized data with both models        
        normal_anon_train = summary_dict[f"model_normal"]["data_anon"]["train"][metric]
        normal_anon_test = summary_dict[f"model_normal"]["data_anon"]["test"][metric]
        anon_anon_train = summary_dict[f"model_anon"]["data_anon"]["train"][metric]
        anon_anon_test = summary_dict[f"model_anon"]["data_anon"]["test"][metric]
        data = {
            "Data used for training": ["Normal"] * len(normal_anon_train + normal_anon_test)
            + ["Anonymized"] * len(anon_anon_train + anon_anon_test),
            metric.capitalize(): normal_anon_train + normal_anon_test + anon_anon_train + anon_anon_test,
            "Split": ["Train"] * len(normal_anon_train)
            + ["Test"] * len(normal_anon_test)
            + ["Train"] * len(anon_anon_train)
            + ["Test"] * len(anon_anon_test),
        }
        df = pd.DataFrame(data)

        plt.figure(figsize=(10, 8))
        sns.boxplot(data=df, x="Data used for training", y=metric.capitalize(), hue="Split", dodge=True)
        if not np.any(df[metric.capitalize()] > 1):
            plt.ylim(0.0, 1.0)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Split", fontsize=16)
        plt.ylabel(metric.capitalize(), fontsize=16)
        plt.legend(title="Dataset used for training", fontsize=14, title_fontsize=16)
        plt.title(f"{metric.capitalize()} comparison for anonymized data between\nmodel trained on normal data and model trained on anonymized data", fontsize=18)
        plt.savefig(f"{output_dir}boxplot_{metric}.png")
        plt.close()

        # Performance on normal data
        normal_normal_train = summary_dict[f"model_normal"]["data_normal"]["train"][metric]
        normal_normal_test = summary_dict[f"model_normal"]["data_normal"]["test"][metric]
        data = {
            "Data used for training": ["Normal"] * len(normal_normal_train + normal_normal_test),
            metric.capitalize(): normal_normal_train + normal_normal_test,
            "Split": ["Train"] * len(normal_normal_train)
            + ["Test"] * len(normal_normal_test)
        }
        df = pd.DataFrame(data)

        plt.figure(figsize=(6, 8))
        sns.boxplot(data=df, x="Data used for training", y=metric.capitalize(), hue="Split", dodge=True)
        # plt.ylim(0.9, 1.0)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Split", fontsize=16)
        plt.ylabel(metric.capitalize(), fontsize=16)
        plt.legend(title="Dataset used for training", fontsize=14, title_fontsize=16)
        plt.title(f"{metric.capitalize()} for model trained and evaluated on normal data", fontsize=18)
        plt.savefig(f"{output_dir}boxplot_normal_{metric}.png")
        plt.close()




def prepare_run(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)
    task = config["task"]
    dataset_name = config["dataset_name"]

    if not os.path.exists(f"data/{task}/{dataset_name}/audiofiles_anonymized"):
        model = load_model()
        anonymize(task, dataset_name, model)

    get_metadata(
        task=task,
        dataset_name=dataset_name,
        label_column_name=config["label_column_name"],
        undersampling=config["undersampling"],
    )

    metadata = prepare_data(config)

    return config, metadata

def run(config, repetition, metadata):
    # model_datasetversion_split_output
    normal_normal_train_output, normal_normal_test_output, normal_anon_train_output, normal_anon_test_output = train(
        config, train_on="normal", repetition=repetition, metadata=metadata
    )
    wandb.finish()
    anon_normal_train_output, anon_normal_test_output, anon_anon_train_output, anon_anon_test_output = train(
        config, train_on="anonymized", repetition=repetition, metadata=metadata
    )
    wandb.finish()

    return {
        "model_normal": {
            "data_normal": {"split_train": normal_normal_train_output, "split_test": normal_normal_test_output},
            "data_anon": {"split_train": normal_anon_train_output, "split_test": normal_anon_test_output}
        },
        "model_anon": {
            "data_normal": {"split_train": anon_normal_train_output, "split_test": anon_normal_test_output},
            "data_anon": {"split_train": anon_anon_train_output, "split_test": anon_anon_test_output}
        }
    }


if __name__ == "__main__":
    main()

    # sumary_dict_path = "/mnt/f/Projects/Janneke/asr-things/models/gender_recognition/gender_recognition_repetitions_metrics.json"
    # with open(sumary_dict_path, "r") as file:
    #     summary_dict = json.load(file)
    # analyse_output_2(output_dir="/mnt/f/Projects/Janneke/asr-things/models/gender_recognition/", summary_dict=summary_dict)
