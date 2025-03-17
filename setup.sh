# This setup follows the steps described in the speaker-anonymization repository
# Make sure you have activated the correct environment before running this script. Python must be at version 3.10 or greater. eg use: "conda create -n asr python=3.10"
# Make sure that you are okay with other repositories being cloned and installed in the folder above your ASR repository


# First check if you are in the correct directory
if ! pwd | grep -q '/ASR$'; then
	echo "Please run this script from the ASR git repository directory"
	exit 1
fi

echo "This script will download the necessary files and install requirements for the speaker-anonymization repository"

# Install some dependencies manually
pip install -r requirements.txt


# Clone the SALT repository
cd ..
git clone https://github.com/BakerBunker/SALT.git
cd SALT/assets
wget https://github.com/BakerBunker/SALT/releases/download/1.0.0/librispeech-pack.zip
unzip librispeech-pack.zip
cd ../..

# Setup dataset for emotion recognition
# Download Crema dataset
# https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/
url="https://www.kaggle.com/datasets/ejlok1/cremad"
mkdir -p data/crema_d/audiofiles
echo "Please download the dataset from: $url and place the unzipped audiofiles in $(pwd)/data/emotion_recognition/crema_d/audiofiles"
read -p "Have you downloaded the dataset? (y/n): " response

if [[ "$response" == "y" || "$response" == "Y" ]]; then
    echo "Well done!"
else
    echo "Please download the dataset before continuing."
    exit 1
fi

# Setup dataset for gender recognition
url="https://www.kaggle.com/datasets/ogechukwu/voice/data"
mkdir -p data/crema_d/audiofiles
echo "Please download the dataset from: $url and place the unzipped audiofiles in the folder one_sentence/one_sentence in $(pwd)/data/gender_recognition/bvc_one_sentence/audiofiles"
read -p "Have you downloaded the dataset? (y/n): " response

if [[ "$response" == "y" || "$response" == "Y" ]]; then
    echo "Well done!"
else
    echo "Please download the dataset before continuing."
    exit 1
fi

echo "Setup complete"
p=$(pwd)
printf 'Please copy and paste the following line to the file ~/.bashrc (nano ~/.bashrc): \n export PYTHONPATH="$PYTHONPATH:%s/SALT"\n' "$p"
