# ASR
This repository was created for the project in the course (Automatic) Speech Recognition of the Radboud University.

Authors are Naomi Deenen, Marieke van Vreeswijk, Laura Stitzel, and Janneke Nouwen

## Setup Instructions
We have provided a bash script to perform the necessary steps to run the code in this repository. Follow the steps below.
1. **Create / activate an environment**

	Make sure you have activated the correct environment before running this script. Python must be at version 3.10 or greater. eg use: `conda create -n asr python=3.10`

2. **Check your working directory**

	Make sure that you are okay with other repositories being cloned in the folder **above** your ASR repository

2. **Run setup**

	Run setup.sh: `bash setup.sh`
	This script will:
	- clone the SALT repository into the parent directory of your ASR repository.
	- install necessary dependencies
	- download the necessary data.

4. **Complete setup**

	After completing the setup, the script will prompt you to add the SALT directory to your `PYTHONPATH`. Please do this.
	Run `source ~/.bashrc` to activate the changes

By following these steps, you will have successfully set up the environment for our project that uses the SALT repository.