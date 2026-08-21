from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI as ChatModel
from dotenv import load_dotenv
from pathlib import Path

from DocProcessor import DocProcessor

if __name__ == "__main__":
    api_key_path = Path(".secrets/OpenAI_API_Keys.secrets").expanduser()
    load_dotenv(api_key_path)  # Load the API Keys 

    llm = ChatModel(model="gpt-5-nano", temperature=0.1, max_tokens=2000)
    db = InMemoryVectorStore(embedding=OpenAIEmbeddings(model="text-embedding-3-small"))

    docLoader = DocProcessor(folderPath=".secrets", fileType=["pdf", "txt"], db=db)
    docLoader.loadAndStoreDocsInFolder()

    # print(llm.invoke("Hello, how are you?").content)