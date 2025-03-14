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
    echo "Well done!"
else
    echo "Please download the dataset before continuing."
    exit 1
fi

# # Download IEMOCAP dataset
# wget https://huggingface.co/datasets/Ar4ikov/iemocap_audio_text/blob/main/data/train-00000-of-00003-9213b91aae6dc76d.parquet

echo "Setup complete!"