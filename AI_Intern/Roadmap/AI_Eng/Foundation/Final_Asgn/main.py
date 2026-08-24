from langchain_openai import ChatOpenAI as ChatModel
from dotenv import load_dotenv
from pathlib import Path
from os import getenv
from hashlib import sha256

from DocProcessor import Parser
from DataBase import DataBase
from RAG import RAG

def get_document_id(file_path: str) -> str:
    '''Generate an Id for each document to avoid uploading stale copies'''
    data = Path(file_path).read_bytes()
    return sha256(data).hexdigest()

def init() -> None:
    global files, filePaths, parser, db, llm, rag

    api_keys_path = Path(".secrets/api_keys.secrets")
    load_dotenv(api_keys_path)
    openAI_api_keys = getenv("OPENAI_API_KEY")

    files = [
        "FSoft_Regulation-Suppliers-Third-Parties",
        "FSoft_Human_Rights_Policy",
        "Policy_Employee-Personal-Data-Protection",
        "Policy_Personal-Data-Privacy"
    ]

    filePaths = [".secrets/" + file + ".pdf" for file in files]

    parser = Parser()
    db = DataBase()
    llm = ChatModel(
        api_key=openAI_api_keys,
        model="gpt-5-nano"
    )
    rag = RAG(db=db, llm=llm)

    print("Init successfully\n")

def init_data(reInit: bool=True) -> None:
    for filePath in filePaths:
        document_id = get_document_id(filePath)

        if not reInit and db.is_document_exists(document_id=document_id):
            print("Load ", filePath, " successfully")
            print("Upload ", filePath, " successfully\n")
            continue

        if reInit:
            db.deleteDoc(document_id=document_id)

        result = parser.loadDoc(filePath=filePath)
        chunks = parser.chunkDocHardCoded(result=result)

        print("Load ", filePath, " successfully")

        for chunk in chunks:
            chunk.document_id = document_id

        db.addDocs(docs=chunks)
        print("Upload ", filePath, " successfully\n")

    print("Init data successfully\n")

if __name__ == "__main__":
    init()
    init_data(False)

    while True:
        query = input("[Query]: ")
        if query.strip().lower() == "quit":
            break
        answer = rag.answer(query=query, limit=5)
        print("[LLM Answer]: ", answer, "\n")

    print("Program finished successfully")