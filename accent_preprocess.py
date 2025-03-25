import os
import pandas as pd
import re
from pydub import AudioSegment

input_dir = 'data/accent_recognition/speech_accent_archive/audiofiles_original'
output_dir = 'data/accent_recognition/speech_accent_archive/audiofiles'
audio_list = os.listdir(input_dir)

# Create DataFrame
df = pd.DataFrame()
df['speech'] = audio_list
# Extract labels (accents) from filenames
df['labels'] = [re.sub(r'\d+\.mp3$', '', audio) for audio in audio_list]
class_names = {'english', 'spanish'}
# Filter to keep only the selected class labels
filtered_df = df[df['labels'].isin(class_names)]

# Loop over filtered dataframe to slice every audio file and save it.
for i, value in filtered_df.iterrows():
    input_file = input_dir + '/' + value['speech']
    audio = AudioSegment.from_file(input_file)

    # Define slice length (5 seconds)
    slice_length = 5 * 1000  # 5 seconds in milliseconds
    num_slices = len(audio) // slice_length + (1 if len(audio) % slice_length != 0 else 0)

    # Slice and save sliced audio 
    for slice_i in range(num_slices):
        start_time = slice_i * slice_length
        end_time = min((slice_i + 1) * slice_length, len(audio))
        slice_audio = audio[start_time:end_time]

        output_path = os.path.join(output_dir,
                                   f"{input_file.split('/')[6][:-4]}_{value['labels']}_slice{slice_i + 1}.wav")

        # Do not save if the audio is shorter than 4 seconds
        if end_time - start_time > 4000:
            slice_audio.export(output_path, format="wav")

print("All slices saved successfully!")