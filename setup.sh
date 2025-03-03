# This setup follows the steps described in the speaker-anonymization repository
# Make sure you have activated the correct environment before running this script
# Make sure that you are okay with other repositories being cloned and installed in the folder above your ASR repository



# First check if you are in the correct directory
if ! pwd | grep -q '/ASR$'; then
	echo "Please run this script from the ASR git repository directory"
	exit 1
fi

echo "This script will download the necessary files and install requirements for the speaker-anonymization repository"

cd ..

git clone --recurse-submodules https://github.com/DigitalPhonetics/speaker-anonymization.git

# Download models
mkdir models
wget https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/embedding_function.pt -P models/embedding_function.pt
wget https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/embedding_gan.pt -P models/embedding_gan.pt
wget https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/aligner.pt -P models/aligner.pt
wget https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/ToucanTTS_Meta.pt -P models/ToucanTTS_Meta.pt
wget https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/Avocodo.pt -P models/Avocodo.pt
wget https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v2.0/asr_branchformer_tts-phn_en.zip -P models/asr_branchformer_tts-phn_en.zip

# Install requirements
pip install -r speaker-anonymization/requirements.txt
pip install typeguard==2.13.3 # to avoid error in the speaker-anonymization repository

# Download MLS and CommonVoice corpora
# Note that we expect that you have CommonVoice version 16.1 or higher for all languages.
mkdir corpora
mkdir corpora/MLS
mkdir corpora/CV
wget https://dl.fbaipublicfiles.com/mls/mls_portuguese.tar.gz -P corpora/MLS

echo "Please download the CommonVoice corpus from https://commonvoice.mozilla.org/en/datasets and place it in corpora/CV"
echo "Continue? (y/n)"
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo "Continuing..."
else
    echo "Exiting..."
    exit 1
fi

# Prepare the data information in kaldi format
# python run_prepare_data.py --mls_path <path-to-MLS-corpus> --cv_path <path-to-CV-corpus> --output_path <path-to-output-files>
































# Set the paths for CUDA and MKL. Change these to match your system
# cuda_path="/usr/lib/cuda"
# # mkl_root=$(dirname $(dirname $(find / -name libmkl_rt.so 2>/dev/null | head -n 1)))  # apt-get install intel-mkl-full -y

# # First check if you are in the correct directory
# if ! pwd | grep -q '/ASR$'; then
# 	echo "Please run this script from the ASR git repository directory"
# 	exit 1
# fi

# echo "This script will download the necessary files and install requirements for the speaker-anonymization repository"

# # Install some dependencies manually
# conda install numpy
# conda install gxx_linux-64
# conda install pyworld

# # Clone repository from the paper
# echo "*** Cloning speaker-anonymization repository ***"
# cd ..
# git clone --recurse-submodules https://github.com/DigitalPhonetics/speaker-anonymization.git
# cd speaker-anonymization
# git checkout gan_embeddings
# sed -i 's/sklearn/scikit-learn/' IMSToucan/requirements.txt  # replace sklearn with scikit-learn in requirements.txt

# # Download models
# echo "*** Downloading models ***"
# mkdir models
# cd models
# for file in anonymization asr tts; do
#     wget https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v1.2/${file}.zip
#     unzip ${file}.zip
#     rm ${file}.zip
# done

# # Install challenge framework (to use for evaluation like in the repo)
# echo "*** Installing challenge framework ***"

# cd ..
# rm -rf Voice-Privacy-Challenge-2020
# git clone --recurse-submodules https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2020.git
# cd Voice-Privacy-Challenge-2020
# pip install -r requirements.txt

# # ./install.sh
# # this step will download and install Kaldi, and might lead to complications. Additionally, make sure that you are running the install script on a device with access to GPUs and CUDA.

# # Install the requirements for the speaker-anonymization repository
# echo "*** Installing requirements for speaker-anonymization ***"
# cd ..
# pwd
# pip install -r requirements.txt

# # Install Kaldi
# echo "*** Installing Kaldi ***"
# cd Voice-Privacy-Challenge-2020

# echo 'Building Kaldi tools'
# cd kaldi/tools
# extras/check_dependencies.sh || exit 1
# make -j $nj || exit 1

# echo 'Building Kaldi src'
# cd ../src
# ./configure --shared --cudatk-dir=${cuda_path} --with-cudadecoder=no || exit 1
# make clean || exit 1
# make depend -j $nj || exit 1
# make -j $nj || exit 1

# cd ../../..

# echo "*** Downloidng and preparing data ***"
# setup_scripts/run_download_data.sh
# setup_scripts/run_prepare_data.sh

# # add anonymization/WGAN to python path
# export PYTHONPATH=$(pwd)/anonymization/WGAN:$PYTHONPATH

# echo "Setup complete"