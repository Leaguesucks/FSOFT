sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install -y curl git unzip xz-utils zip libglu1-mesa cmake

sudo apt install python3
sudo apt install python3-venv
sudo apt install python3-pip
sudo apt install uvicorn

python3 -m venv .venv
source .venv/bin/activate

pip install fastapi uvicorn pydantic
pip install langchain langchain-core langchain-community langchain-openai
pip install dotenv 
pip install llama-cloud qdrant-client
pip install httpx tavily-python

cd agent
flutter pub add http
cd ..
