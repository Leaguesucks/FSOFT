import httpx, os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from LLM.Query.QueryResolver import QueryResolver
from LLM.Query.QueryParallelizer import QueryPlanner
from LLM.Document.Summarizer import Candidate_Summary

from Tools.CodeExecutor import Language

class Model:
    """Holding the LLM model"""
    EXECUTION_SERVICE_IP = "http://127.0.0.1:8000/execute"

    def __init__(self):
        load_dotenv(".secrets/api_keys.secrets")

        self.llm_heavy_second = ChatOpenAI(
            model="gpt-5.6-luna",
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.llm_heavy = ChatGroq(model="openai/gpt-oss-120b")
        self.llm_light = ChatGroq(model="openai/gpt-oss-20b")

        self.query_resolver = QueryResolver()     
        self.decomposer = QueryPlanner()

        self.summarizer = self.llm_heavy_second.with_structured_output(Candidate_Summary)

    def execute_python(self, code: str):
        '''Execute a python code and return the result'''
        response = httpx.post(
            self.EXECUTION_SERVICE_IP,
            json={
                "language": Language.PYTHON.value,
                "code": code
            },
            timeout=10
        )

        response.raise_for_status()
        return response.json()