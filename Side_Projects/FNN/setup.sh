# Don't forget to grant permission using chmod +x setup.sh
sudo apt update
sudo apt install build-essential
sudo apt install libopencv-dev
sudo apt install python3-venv
sudo apt install python3-pip

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --upgrade matplotlib numpy
pip install opencv-python
pip install torch