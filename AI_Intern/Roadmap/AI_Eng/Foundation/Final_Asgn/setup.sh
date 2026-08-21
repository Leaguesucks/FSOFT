# Remember to grant permissions to the setup.sh file using the command: chmod +x setup.sh on Linux or macOS.
# Run ./setup.sh on Linux or ./setup.bat on Windows to install dependencies and set up the environment.

sudo apt update
sudo apt install python3
sudo apt install python3-venv
sudo apt install python3-pip
sudo apt install pdftotext # Always convienient

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r dependencies.txt
# pip install --no-cache-dir docling --extra-index-url https://download.pytorch.org/whl/cpu # NOT RECOMMEND: Heavy strain on local machine