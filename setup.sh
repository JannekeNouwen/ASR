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
# and torch must be v2.0 or greater
pip install -r requirements.txt


# Clone the SALT repository
cd ..
git clone https://github.com/BakerBunker/SALT.git
cd SALT/assets
wget https://github.com/BakerBunker/SALT/releases/download/1.0.0/librispeech-pack.zip
unzip librispeech-pack.zip
cd ../../ASR


# add anonymization/WGAN to python path
# export PYTHONPATH=$(pwd)/anonymization/WGAN:$PYTHONPATH

# echo Please add the following line to the file ~/.bashrc: "export PYTHONPATH=\$(pwd)/anonymization/WGAN:'\$PYTHONPATH"


echo "Setup complete"