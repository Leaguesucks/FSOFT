from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType, ScoredPoint
from dotenv import load_dotenv
from os import getenv
from pathlib import Path
from langchain_openai import OpenAIEmbeddings

from DocProcessor import Chunk

class DataBase:
    '''The Vector Database to store relevant documents'''

    def __init__(self):
        qdrant_api_keys_path = Path(".secrets/api_keys.secrets")
        load_dotenv(qdrant_api_keys_path)
        qdrant_api_key = getenv("QDRANT_API_KEY")
        qdrant_endpoint = getenv("QDRANT_CLUSTER_ENDPOINT")
        openAI_api_key = getenv("OPENAI_API_KEY")

        self.client = QdrantClient(
            url=qdrant_endpoint,
            api_key=qdrant_api_key,
            cloud_inference=True,
            timeout=60
        )

        self.COLLECTION_NAME = "Documents"
        self.COLLECTION_SIZE = 1536

        if not self.client.collection_exists(self.COLLECTION_NAME):
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.COLLECTION_SIZE,
                    distance=Distance.COSINE
                )
            )

        self.client.create_payload_index(
            collection_name=self.COLLECTION_NAME,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD
        )

        self.embedding = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openAI_api_key
        )

    def addDocs(self, docs: list[Chunk], batch_size: int=20) -> None:
        '''Embed and add documents to the database'''
        if not docs:
            return

        contents = [doc.content for doc in docs]
        vectors = self.embedding.embed_documents(contents)

        points = []
        for doc, vector in zip(docs, vectors):
            points.append(
                PointStruct(
                    id=doc.id,
                    vector=vector,
                    payload={
                        "document_name": doc.document_name,
                        "document_id": doc.document_id,
                        "title": doc.title,
                        "heading_type": doc.heading_type,
                        "level": doc.level,
                        "owner": doc.owner,
                        "content": doc.content,
                        "full_content": doc.full_content,
                        "pages": doc.pages
                    }
                )
            ) 

        # Upload in batches to avoid timeout error
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=batch
            )

    def deleteDoc(self, document_id: str) -> None:
        '''Delete all chunks belonging to a document using the document's id'''
        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        )

    def deleteChunks(self, ids: list[str]) -> None:
        '''Delete chunks by their id'''
        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=ids
        )

    def is_document_exists(self, document_id: str) -> bool:
        '''Check whether at least one chunk belonging to the document exists'''
        result = self.client.count(
            collection_name=self.COLLECTION_NAME,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            ),
            exact=True
        )

        return result.count > 0

    def search(self, query: str, limit: int=5, document_id: str | None=None) -> list[ScoredPoint]:
        '''Search for chunks relevant to the query'''
        query_vector = self.embedding.embed_query(query)
        query_filter = None

        if document_id is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        )

        return results.points