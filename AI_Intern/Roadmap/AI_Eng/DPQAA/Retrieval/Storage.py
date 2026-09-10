import sys,bm25s, Stemmer
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType, ScoredPoint
from dotenv import load_dotenv
from os import getenv
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from dataclasses import dataclass

from Indexing.Parser import Chunk

@dataclass
class SearchResult:
    id: str
    score: float
    payload: dict

    #debug
    semantic_rank: int=0
    bm25_rank: int=0

class Storage:
    def __init__(self, collection_name: str="Documents"):
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

        self.COLLECTION_NAME = collection_name
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

        self.bm25 = None
        self.bm25_chunks = []
        self.build_bm25_index()

    def build_bm25_index(self) -> None:
        '''Build an in-memory BM25 index from all chunks in Qdrant.'''
        stemmer = Stemmer.Stemmer("english")

        self.bm25_chunks = []
        documents = []

        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            ) 

            for point in points:
                payload = point.payload or {}
                content = payload.get("content", "")

                if not content:
                    continue

                self.bm25_chunks.append(point)
                documents.append(content)

            if offset is None:
                break

        if not documents:
            self.bm25 = None
            return

        corpus_tokens = bm25s.tokenize(
            documents,
            stopwords="en",
            stemmer=stemmer,
            show_progress=True
        )

        self.bm25 = bm25s.BM25(method="lucene")
        self.bm25.index(corpus_tokens)

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
                        "parent_id": doc.parent_id,
                        "root_id": doc.root_id,
                        "content": doc.content,
                        **(
                            {"full_content": doc.full_content}
                            if doc.parent_id is None
                            else {}
                        ),
                        "pages": doc.pages
                    }
                )
            ) 

        # Upload in batches to avoid timeout error
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=batch
            )

        # Rebuild BM25 tokens
        self.build_bm25_index()

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

        # Rebuild BM25 tokens
        self.build_bm25_index()

    def deleteChunks(self, ids: list[str]) -> None:
        '''Delete chunks by their id'''
        if not ids:
            return

        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=ids
        )

        # Rebuild BM25 after deletion
        self.build_bm25_index()

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

    def get_chunk(self, chunk_id: str) -> SearchResult | None:
        '''Retrieved a chunk by their ID'''
        results = self.client.retrieve(
            collection_name=self.COLLECTION_NAME,
            ids=[chunk_id],
            with_payload=True
        )

        if not results:
            return None

        result = results[0]

        return SearchResult(
            id=str(result.id),
            score=0.0,
            payload=result.payload or {}
        )

    def group_by_root(self, results: list[SearchResult]) -> dict[str, list[SearchResult]]:
        '''Group retrieved chunks by their root chunk'''
        groups: dict[str, list[SearchResult]] = {}

        for result in results:
            root_id = result.payload.get("root_id")

            if root_id is None:
                continue

            groups.setdefault(root_id, []).append(result)

        return groups

    def get_vectors(self, ids: list[str]) -> dict[str, np.array]:
        """Retrieve vectors from Qdrant for the given document's point ids"""
        results = self.client.retrieve(
            collection_name=self.COLLECTION_NAME,
            ids=ids,
            with_payload=False,
            with_vectors=True
        )

        return {
            str(result.id): np.asarray(result.vector, dtype=np.float32)
            for result in results
        }

    def similarity_cosine(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        '''Return the cosine similarity between two vectors'''
        mag1 = np.linalg.norm(vec1)
        mag2 = np.linalg.norm(vec2)

        if mag1 == 0.0 or mag2 == 0.0:
            return 0.0

        return float(np.dot(vec1, vec2) / (mag1 * mag2))
    
    def search_semantic(self, query: str, limit: int=10, document_id: str | None=None) -> list[SearchResult]:
        '''Search for chunks relevant to the query.'''
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

        return [
            SearchResult(
                id=str(result.id),
                score=float(result.score),
                payload=result.payload
            )
            for result in results.points
        ]

    def search_bm25(self, query: str, limit: int=10, document_id: str | None = None) -> list[SearchResult]:
        if self.bm25 is None:
            return []

        query_tokens = bm25s.tokenize(
            query,
            stopwords="en",
            stemmer=Stemmer.Stemmer("english")
        )

        results, scores = self.bm25.retrieve(
            query_tokens,
            k=limit
        )

        search_results = []

        for index, score in zip(results[0], scores[0]):
            point = self.bm25_chunks[index]

            if document_id is not None:
                if point.payload.get("document_id") != document_id:
                    continue

            search_results.append(
                SearchResult(
                    id=str(point.id),
                    score=float(score),
                    payload=point.payload
                )
            )

        return search_results

    def search_hybrid(self, query: str, k: float=60.0, limit: int=5, document_id: str | None = None) -> list[SearchResult]:
        '''Hybrid search using RRF algorithm'''

        # Dense retrieval
        semantic_results = self.search_semantic(query=query, limit=20, document_id=document_id)
        bm25_results = self.search_bm25(query=query, limit=20, document_id=document_id)

        # Sanity sort
        semantic_results = sorted(semantic_results, key=lambda result: result.score, reverse=True)
        bm25_results = sorted(bm25_results, key=lambda result: result.score, reverse=True)

        semantic_ranks = {
            result.id: rank
            for rank, result in enumerate(semantic_results, start=1)
        }

        bm25_ranks = {
            result.id: rank 
            for rank, result in enumerate(bm25_results, start=1)
        }

        candidates = {}

        for result in semantic_results:
            candidates[result.id] = result

        for result in bm25_results:
            candidates[result.id] = result

        results = []
        for chunk_id, result in candidates.items():
            score = 0.0

            if chunk_id in semantic_ranks:
                semantic_rank = semantic_ranks[chunk_id]
                score += 1.0 / (k + float(semantic_rank))

            if chunk_id in bm25_ranks:
                bm25_rank = bm25_ranks[chunk_id]
                score += 1.0 / (k + float(bm25_rank))

            results.append(
                SearchResult(
                    id=result.id,
                    score=score,
                    payload=result.payload,
                    semantic_rank=semantic_rank,
                    bm25_rank=bm25_rank
                )
            )

        results.sort(key=lambda result: result.score, reverse=True)

        return results[:limit]

    def search_mmr(self, docs: list[SearchResult], landa: float=0.5, limit: int=10) -> list[SearchResult]:
        '''Re-rank and retrieved the sorted documents using MMR
            NOTE: Internal uses only
        '''
        documents = docs.copy()
        documents.sort(key= lambda result: result.score, reverse=True)

        vectors = self.get_vectors(document.id for document in documents)

        S: list[SearchResult] = []

        while documents and len(S) < limit:
            best_mmr, best_i = -sys.float_info.max, 0

            for i, document in enumerate(documents):
                sims = [
                    self.similarity_cosine(
                        vectors[s.id],
                        vectors[document.id]
                    )
                    for s in S
                ]

                max_sim = max(sims) if sims else 0.0
                mmr = landa * document.score - (1 - landa) * max_sim

                if (mmr > best_mmr):
                    best_mmr = mmr
                    best_i = i

            S.append(documents.pop(best_i))

        return S


    def search(self, query: str,
               k=60.0, landa=0.5,
               top_k=5,
               document_id: str | None = None
    ) -> list[SearchResult]:
        hybrid_results = self.search_hybrid(
            query=query,
            limit=20,
            document_id=document_id
        )

        return self.search_mmr(
            docs=hybrid_results,
            limit=top_k
        )

                