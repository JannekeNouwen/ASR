# THIS IS OLD, DO NOT USE


# import pandas as pd
# from speechbrain.inference.interfaces import foreign_class

# from emotion_recognition_wav2vec import custom_interface
# from emotion_recognition_wav2vec.custom_interface import CustomEncoderWav2vec2Classifier
# from tqdm import tqdm
# input_path = "/home/janneke/scripts/personal/ASR-stuff/data/emotions_dataset/Crema/1001_DFA_ANG_XX.wav"
# source = custom_interface.__file__

# emotions in the CREMA-D dataset

# emotions in the IEMOCAP dataset
# anger, happiness, excitement, sadness, frustration, fear, surprise, other and neutral state


# def main():
#     crema_data = pd.read_csv("./data/crema_d/crema.csv")
#     print(crema_data.head())
#     classifier = load_model()

#     # Create a subset for testing
#     crema_data = crema_data.sample(20)

#     predictions = []
#     for i, row in tqdm(crema_data.iterrows(), total=len(crema_data)):
#         emotion = row["emotion"]
#         input_path = f"./data/crema_d/audiofiles/{row['speaker']}_{row['sentence']}_{emotion.upper()}_{row['emotion_level']}.wav"
#         predictions.append(predict(classifier, input_path))

#     crema_data["predicted_emotion"] = predictions
#     crema_data.to_csv("./data/crema_d/crema_predictions.csv")


# def load_model():
#     classifier: CustomEncoderWav2vec2Classifier = foreign_class(
#         source=source.replace("/custom_interface.py", ""),
#         pymodule_file="custom_interface.py",
#         classname="CustomEncoderWav2vec2Classifier",
#     )
#     return classifier


# def predict(classifier: CustomEncoderWav2vec2Classifier, input_path):
#     out_prob, score, index, text_lab = classifier.classify_file(input_path)
#     return text_lab


# if __name__ == "__main__":
#     main()
