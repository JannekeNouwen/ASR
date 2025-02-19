# This setup follows the steps described in the speaker-anonymization repository
# Make sure you have activated the correct environment before running this script

# First check if you are in the correct directory
if ! pwd | grep -q '/ASR$'; then
	echo "Please run this script from the ASR git repository directory"
	exit 1
fi

echo "This script will download the necessary files and install requirements for the speaker-anonymization repository"

# Clone repository from the paper
echo "*** Cloning speaker-anonymization repository ***"
cd ..
git clone --recurse-submodules https://github.com/DigitalPhonetics/speaker-anonymization.git
cd speaker-anonymization

# Download models
echo "*** Downloading models ***"
mkdir models
cd models
for file in anonymization asr tts; do
    wget https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v1.2/${file}.zip
    unzip ${file}.zip
    rm ${file}.zip
done

# Install challenge framework (to use for evaluation like in the repo)
echo "*** Installing challenge framework ***"
cd ../..
git clone git@github.com:Voice-Privacy-Challenge/Voice-Privacy-Challenge-2020.git
cd Voice-Privacy-Challenge-2020
./install.sh
# this step will download and install Kaldi, and might lead to complications. Additionally, make sure that you are running the install script on a device with access to GPUs and CUDA.

# Install the requirements for the speaker-anonymization repository
echo "*** Installing requirements for speaker-anonymization ***"
cd ../speaker-anonymization
pip install -r requirements.txt
