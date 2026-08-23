from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter as TextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from pathlib import Path

class DocProcessor:
    '''Class to process documents by loading and splitting them into chunks, then upload it to vector store
    @depreciated: This class is deprecated and only kept for legacy purposes. Please use the ModernDocProcessor
    '''
    def __init__(self, folderPath: str, fileType: list[str], db: InMemoryVectorStore):
        self.folderPath = folderPath
        self.fileType = fileType
        self.db = db
        self.unique_id_counter = 0 # Assign each document a unique Id

    def loadDoc(self, file: str, fileType: str) -> list[Document]:
        '''
        Load and return documents from the specified file path and type.
        '''

        if fileType == "pdf":
            loader = PyPDFLoader(file_path=file, mode="single")
        elif fileType == "txt":
            loader = TextLoader(file_path=file, encoding="utf-8")
        else:
            raise ValueError("Unsupported file type. Please use 'pdf' or 'txt'.")
        
        documents = loader.load()
        return documents

    def loadAndStoreDocsInFolder(self) -> None:
        '''
        Load and return documents from all files in the specified folder path with the given file type then load them into the vector store.
        Also partition the documents into smaller chunks and store them to the database.
        '''
        folder = Path(self.folderPath).expanduser()

        if not folder.exists():
            raise FileNotFoundError(f"The folder path \'{self.folderPath}\' does not exist.")

        if not folder.is_dir():
            raise NotADirectoryError(f"The path \'{self.folderPath}\' is not a directory.")

        for file in folder.iterdir():
            if not file.is_file():
                continue

            fileType = file.suffix.lower().lstrip('.')

            if fileType in self.fileType:
                document = self.loadDoc(str(file), fileType)
                if document:
                    document[0].id = str(self.unique_id_counter)
                    self.db.add_documents(document)
                    self.splitRecursiveLy(document[0], chunksize=1000, chunk_overlap=200)
                    self.unique_id_counter += 1
                    print(document[0].metadata["source"], ": ", file, ": ", len(document[0].page_content))


    def splitDoc(self, document: Document, chunking_mode: str = "recursive", chunk_size: int = 1000, 
                chunk_overlap: int = 200, delim: str = "\n\n") -> list[Document]:
        '''
        Split the loaded documents into smaller chunks based on the specified chunking mode, size, and overlap.
        The chunking_mode parameter currently supports only "recursive" or "fixed_size" splitting. The chunk_size and chunk_overlap parameters 
        define the size of each chunk and the overlap between consecutive chunks, respectively.
        The delim parameter specifies how to recursively chunk the documents.
        '''
        if chunking_mode == "recursive":
            text_splitter = TextSplitter(separators=[delim], keep_separator=False, chunk_overlap=chunk_overlap, chunk_size=chunk_size)
        elif chunking_mode == "fixed_size":
            text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            raise ValueError("Unsupported chunking mode. Please use 'recursive'.")
        
        split_documents = text_splitter.split_documents([document])
        return split_documents

    def splitRecursiveLy(self, document: Document, chunksize=1000,chunk_overlap=200) -> None:
        '''
        Automatically split the loaded documents into paragraphs and sentences recursively. Then load them to the database.
        '''
        paragraphs = self.splitDoc(document, chunking_mode="recursive", delim="\n", chunk_size=chunksize, chunk_overlap=chunk_overlap)

        paragraph_counter = 0
        for paragraph in paragraphs:
            paragraph.id = document.id + "." + str(paragraph_counter)
            paragraph.metadata["type"] = "paragraph"
            paragraph.metadata["parent_id"] = document.id
            self.db.add_documents([paragraph])
            lines = self.splitDoc(paragraph, chunking_mode="recursive", delim=".", chunk_size=chunksize, chunk_overlap=chunk_overlap)
            paragraph_counter += 1

            line_counter = 0
            for line in lines:
                line.id = paragraph.id + "." + str(line_counter)
                line.metadata["type"] = "line"
                line.metadata["parent_id"] = paragraph.id
                self.db.add_documents([line])
                line_counter += 1