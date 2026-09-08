sudo apt update
sudo apt install build-essential
sudo apt install python3-venv
sudo apt install python3-pip

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests warcio