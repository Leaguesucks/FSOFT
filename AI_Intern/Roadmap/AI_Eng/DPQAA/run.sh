#!/bin/bash

BACKEND_DIR="$HOME/FSOFT/AI_Intern/Roadmap/AI_Eng/DPQAA"
FLUTTER_DIR="$HOME/FSOFT/AI_Intern/Roadmap/AI_Eng/DPQAA/agent"
VENV="$HOME/FSOFT/AI_Intern/Roadmap/AI_Eng/DPQAA/.venv"

echo "Starting FastAPI..."

cd "$BACKEND_DIR"
source "$VENV/bin/activate"

# python3 -m uvicorn main:app --reload
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

BACKEND_PID=$!

kill $BACKEND_PID