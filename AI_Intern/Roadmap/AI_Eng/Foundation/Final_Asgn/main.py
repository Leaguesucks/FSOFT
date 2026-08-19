from xml.dom.minidom import Document

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter as TextSplitter
from langchain_openai import OpenAIEmbeddings as Embeddings
from langchain_openai import ChatOpenAI as ChatModel
from langchain_core.vectorstores import InMemoryVectorStore as VectorStore
import os, getpass

llm = ChatModel(model="gpt-5-nano", temperature=0.1, max_tokens=2000)
embeddings = Embeddings(model="text-embedding-3-small")
vectorStore = VectorStore(embeddings=embeddings)

def loadDoc(filePath: str, fileType: str) -> list[Document]:
    '''
    Load and return documents from the specified file path and type.'''
    if fileType == "pdf":
        loader = PyPDFLoader(file_path=filePath, mode="single")
    elif fileType == "txt":
        loader = TextLoader(filePath)
    else:
        raise ValueError("Unsupported file type. Please use 'pdf' or 'txt'.")
    
    documents = loader.load()
    return documents

def splitDoc(documents: list[Document], chunking_mode: str = "recursive", chunk_size: int = 1000, 
             chunk_overlap: int = 200, delim: str = "\n\n") -> list[Document]:
    '''
    Split the loaded documents into smaller chunks based on the specified chunking mode, size, and overlap.
    The chunking_mode parameter currently supports only "recursive" or "fixed_size" splitting. The chunk_size and chunk_overlap parameters 
    define the size of each chunk and the overlap between consecutive chunks, respectively.
    The delim parameter specifies how to recursively chunk the documents.
    '''
    if chunking_mode == "recursive":
        text_splitter = TextSplitter(separators=delim, chunk_overlap=chunk_overlap)
    elif chunking_mode == "fixed_size":
        text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError("Unsupported chunking mode. Please use 'recursive'.")
    
    split_documents = text_splitter.split_documents(documents)
    return split_documents

def storeEmbeddings(documents: list[Document], vectorStore: VectorStore) -> list[str]:
    '''
    Embed and store the provided documents in the specified vector store.
    '''
    return vectorStore.add_documents(documents)

def retrieveRelevantDocs(query: list[str], vectorStore: VectorStore, top_k: int = 5) -> list[Document]:
    '''
    Retrieve the most relevant documents from the vector store based on the provided query.
    The top_k parameter specifies how many of the most relevant documents to return.
    '''
    retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    return retriever.batch([query])