# This setup follows the steps described in the speaker-anonymization repository
# Make sure you have activated the correct environment before running this script (python=3.8)
# Make sure that you are okay with other repositories being cloned and installed in the folder above your ASR repository

# After running this script, do the following:
# 1. rename the speaker-anonymization folder to speaker_anonymization and add it to your python path. Add this line to your .bashrc file (adapt the path to your own):
  # export PYTHONPATH="${PYTHONPATH}:/home/janneke/Documents/Master/ASR-project/speaker_anonymization"
  # Now run source ~/.bashrc to activate the changes
# 2. Go to speaker_anonymization/models/anonymization/gan/settings.json and remove the comma at the end of line 5

# Okay great. No further reading neccessary. Just run the script and you're good to go. (Unless you ran into errors, then check this script for notes or message Janneke)

# Set the paths for CUDA. Change these to match your system
cuda_path="/usr/lib/cuda"
# mkl_root=$(dirname $(dirname $(find / -name libmkl_rt.so 2>/dev/null | head -n 1)))  # apt-get install intel-mkl-full -y

# First check if you are in the correct directory
if ! pwd | grep -q '/ASR$'; then
	echo "Please run this script from the ASR git repository directory"
	exit 1
fi

echo "This script will download the necessary files and install requirements for the speaker-anonymization repository"

# Ask the user if they read the instructions at the top of the file. If not, exit.
read -p "Have you read the instructions at the top of this file? [y/n] " -n 1 -r
echo "Great, let's continue. This could take some time. Do not forget to follow the instructions at the top of this file after running this script."

sleep 4

# Install some dependencies manually
conda install numpy
conda install gxx_linux-64
conda install pyworld
pip install pyloudnorm

# Install ESPnet2
sudo apt-get install sox
sudo apt-get install flac
git clone https://github.com/espnet/espnet
cd espnet/tools
rm -f activate_python.sh && touch activate_python.sh
make

cd ../..

# Clone repository from the paper
echo "*** Cloning speaker-anonymization repository ***"
cd ..
git clone --recurse-submodules https://github.com/DigitalPhonetics/speaker-anonymization.git
cd speaker-anonymization
git checkout gan_embeddings
sed -i 's/sklearn/scikit-learn/' IMSToucan/requirements.txt  # replace sklearn with scikit-learn in requirements.txt

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

cd ..
rm -rf Voice-Privacy-Challenge-2020
git clone --recurse-submodules https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2020.git
cd Voice-Privacy-Challenge-2020
pip install -r requirements.txt

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # Or pick the right one for your CUDA version

# ./install.sh
# this step will download and install Kaldi, and might lead to complications. Additionally, make sure that you are running the install script on a device with access to GPUs and CUDA.

# If the script fails after any of the steps below, you can probably ignore it and just continue with whatever you want to do.

# Install the requirements for the speaker-anonymization repository
echo "*** Installing requirements for speaker-anonymization ***"
cd ..
pwd
pip install -r requirements.txt
sudo apt-get install espeak-ng
touch __init__.py  # This makes sure that the folder is recognized as a package so we can import from it

# Install Kaldi
echo "*** Installing Kaldi ***"
cd Voice-Privacy-Challenge-2020

echo 'Building Kaldi tools'
cd kaldi/tools
extras/check_dependencies.sh || exit 1
make -j $nj || exit 1

echo 'Building Kaldi src'
cd ../src
./configure --shared --cudatk-dir=${cuda_path} --with-cudadecoder=no || exit 1
make clean || exit 1
make depend -j $nj || exit 1
make -j $nj || exit 1

cd ../../..

echo "*** Downloidng and preparing data ***"
setup_scripts/run_download_data.sh
setup_scripts/run_prepare_data.sh

echo "Setup complete. Please follow the instructions at the top of the file."
