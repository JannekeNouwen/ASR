# ## does not work yet, so right now done through kaggle
#
#
# import os
# import pandas as pd
# import re
# # from pydub import AudioSegment
#
# path = 'data/accent_recognition/speech_accent_archive/audiofiles'
#
# audio_list = os.listdir(path)
#
# # Create DataFrame
# df = pd.DataFrame()
# df['speech'] = audio_list
#
# # Extract labels (accents) from filenames
# df['labels'] = [re.sub(r'\d+\.mp3$', '', audio) for audio in audio_list]
#
# class_names = {'english','spanish'}
#
# # Filter to keep only the selected class labels
# filtered_df = df[df['labels'].isin(class_names)]
#
# # Load the WAV file
# for i, j in filtered_df.iterrows():
#
#     input_file = path + '/' + j['speech']
#     print(input_file)
#     audio = AudioSegment.from_file(input_file)
#     # Define slice length (5 seconds)
#     slice_length = 5 * 1000  # 5 seconds in milliseconds
#     num_slices = len(audio) // slice_length + (1 if len(audio) % slice_length != 0 else 0)
#
#     # Create output directory
#     output_dir = path + '_sliced'
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Slice and save
#     for i in range(num_slices):
#         start_time = i * slice_length
#         end_time = min((i + 1) * slice_length, len(audio))
#         slice_audio = audio[start_time:end_time]
#
#         output_path = os.path.join(output_dir, f"{input_file.split('/')[6][:-4]}_{j['labels']}_slice{i + 1}.wav")
#         slice_audio.export(output_path, format="wav")
#
# print("All slices saved successfully!")
