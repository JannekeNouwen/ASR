# First check if you are in the correct directory
if ! pwd | grep -q '/ASR$'; then
	echo "Please run this script from the ASR git repository directory"
	exit 1
fi

cd ..

pip install git+https://github.com/speechbrain/speechbrain.git@develop

git clone git@hf.co:speechbrain/emotion-recognition-wav2vec2-IEMOCAP
mv -r emotion-recognition-wav2vec2-IEMOCAP emotion_recognition
export PYTHONPATH=$PYTHONPATH:$(pwd)/emotion_recognition

p=$(pwd)
printf 'Please copy and paste the following line to the file ~/.bashrc (nano ~/.bashrc): \n export PYTHONPATH="$PYTHONPATH:%s/emotion_recognition"\n' "$p"