from fastapi import APIRouter
from pydantic import BaseModel
from enum import Enum

import subprocess
import tempfile
import os

class Language(Enum):
    '''For future development'''
    PYTHON = "python"
    JAVA = "java"
    C = "C"
    CPP = "CPP"

class Request(BaseModel):
    language: Language
    code: str

class Response(BaseModel):
    stdout: str
    stderr: str
    exit_code: int

class Executor:
    def __init__(self):
        self.router = APIRouter(prefix="/execute")
        self.router.post("", response_model=Response)(self.execute)

    def execute(self, request: Request) -> Response:
        if request.language != Language.PYTHON: # Currently only support Python
            return Response(
                stdout="",
                stderr="Unsupported language",
                exit_code=1
            )

        try:
            with tempfile.TemporaryDirectory() as tmp:
                code_file = os.path.join(tmp, "main.py")

                with open(code_file, "w") as f:
                    f.write(request.code)

                result = subprocess.run(
                    ["python3", code_file],
                    capture_output=True, text=True, timeout=5)

                return Response(
                    stdout=result.stdout,
                    stderr=result.stderr,
                    exit_code=result.returncode
                )
        except subprocess.TimeoutExpired:
            return Response(
                stdout="",
                stderr="Execution timed out",
                exit_code=124
            )

    