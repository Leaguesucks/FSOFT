# Don't forget to grant permission using chmod +x setup.sh
sudo apt update
sudo apt install build-essential
sudo apt install libopencv-dev

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install matplotlib numpy
pip install torch