# First check if you are in the correct directory
if ! pwd | grep -q '/ASR$'; then
	echo "Please run this script from the ASR git repository directory"
	exit 1
fi

cd ..

# Download Crema dataset
# https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/
url="https://www.kaggle.com/datasets/ejlok1/cremad"
mkdir -p data/crema_d/audiofiles
echo "Please download the dataset from: $url and place the unzipped audiofiles in $(pwd)/data/crema_d/audiofiles"
read -p "Have you downloaded the dataset? (y/n): " response

if [[ "$response" == "y" || "$response" == "Y" ]]; then
    echo "Proceeding with the setup..."
else
    echo "Please download the dataset before continuing."
    exit 1
fi

# Download IEMOCAP dataset
wget https://huggingface.co/datasets/Ar4ikov/iemocap_audio_text/blob/main/data/train-00000-of-00003-9213b91aae6dc76d.parquet


# Install speechbrain
pip install git+https://github.com/speechbrain/speechbrain.git@develop

# Clone the emotion recognition repository
git clone git@hf.co:speechbrain/emotion-recognition-wav2vec2-IEMOCAP
mv emotion-recognition-wav2vec2-IEMOCAP emotion_recognition_wav2vec
cd emotion_recognition_wav2vec
wget https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP/resolve/main/model.ckpt
wget https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP/blob/main/wav2vec2.ckpt
export PYTHONPATH=$PYTHONPATH:$(pwd)/emotion_recognition

p=$(pwd)
printf 'Please copy and paste the following line to the file ~/.bashrc (nano ~/.bashrc): \n export PYTHONPATH="$PYTHONPATH:%s/emotion_recognition_wav2vec"\n' "$p"