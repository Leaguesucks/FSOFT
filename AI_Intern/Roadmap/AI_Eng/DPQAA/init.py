'''Run this file first to upload the documents'''

from dotenv import load_dotenv
from pathlib import Path
from os import getenv
from hashlib import sha256

from Indexing.Parser import Parser
from Retrieval.Storage import Storage

def get_document_id(file_path: str) -> str:
    '''Generate an Id for each document to avoid uploading stale copies'''
    data = Path(file_path).read_bytes()
    return sha256(data).hexdigest()

def init_data(reInit: bool=True) -> None:
    for file in files:
        filePath = ".secrets/" + file
        document_id = get_document_id(filePath)

        if not reInit and db.is_document_exists(document_id=document_id):
            print("Skip")
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
    global files, parser, db

    with open("Supplies/documents.txt", "r") as rf:
        files = [line.strip() for line in rf if line.strip()]

    parser = Parser()
    db = Storage()

    init_data(False)

    

