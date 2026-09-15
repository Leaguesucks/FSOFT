import sys,bm25s, Stemmer
import numpy as np

import Tools.Math as math

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType, ScoredPoint
from dotenv import load_dotenv
from os import getenv
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from Indexing.Parser import Chunk
from Tools.Math import Stats, Standout_Category

@dataclass
class SearchResult:
    id: str
    payload: dict

    #debug
    semantic_rank: int=0
    bm25_rank: int=0

    bm25_score: float=0.0
    semantic_score: float=0.0
    hybrid_score: float=0.0
    mmr: float=0.0

@dataclass
class BM25SearchResult(SearchResult):
    stat: Stats | None = None
    standout: bool=False

class Storage:
    def __init__(self, collection_name: str="Documents",
                 semantic_threshold: float=0.4,
                 bm25_threshold=2.0):
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

        self.SEMANTIC_THRESHOLD = semantic_threshold
        self.BM25_THRESHOLD = bm25_threshold

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

        self.search_executor = ThreadPoolExecutor(max_workers=2)

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
            semantic_score=0.0,
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
    
    def search_semantic(self, query: str, limit: int=10) -> list[SearchResult]:
        '''Search for chunks relevant to the query.'''
        query_vector = self.embedding.embed_query(query)
        query_filter = None

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
                semantic_score=float(result.score),
                payload=result.payload
            )
            for result in results.points if result.score >= self.SEMANTIC_THRESHOLD
        ]

    def search_bm25(self, query: str, limit: int=10) -> list[BM25SearchResult]:
        '''Return a list of search results using BM25'''

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

            search_results.append(
                BM25SearchResult(
                    id=str(point.id),
                    bm25_score=float(score),
                    payload=point.payload
                )
            )

        standout_result = math.retrieve_stat(
            scores=np.asarray(scores[0], dtype=np.float32)
        )

        if (
                standout_result.z_score >= Standout_Category.WELL_ABOVE.value and
                scores[0][0] >= self.BM25_THRESHOLD
            ):
            search_results[0].standout = True
            search_results[0].stat = standout_result

        return search_results

    def search_hybrid(self, query: str, k: float=60.0, limit: int=5) -> list[SearchResult]:
        '''Hybrid search using RRF algorithm'''
        semantic_future = self.search_executor.submit(
            self.search_semantic,
            query,
            20
        )

        bm25_future = self.search_executor.submit(
            self.search_bm25,
            query,
            20
        )

        semantic_results = semantic_future.result()
        bm25_results = bm25_future.result()
        
        semantic_ranks = {
            result.id: rank
            for rank, result in enumerate(semantic_results, start=1)
        }

        bm25_ranks = {
            result.id: rank 
            for rank, result in enumerate(bm25_results, start=1)
        }

        candidates: dict[str, SearchResult] = {}

        for result in semantic_results:
            candidates[result.id] = result

        for result in bm25_results:
            if result.id not in candidates: 
                candidates[result.id] = result
            else:
                candidates[result.id].bm25_score = result.bm25_score

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
                    payload=result.payload,

                    semantic_rank=semantic_ranks.get(chunk_id, 0),
                    bm25_rank=bm25_ranks.get(chunk_id, 0),

                    semantic_score=result.semantic_score,
                    bm25_score=result.bm25_score,
                    hybrid_score=score
                )
            )

        results.sort(key=lambda result: result.hybrid_score, reverse=True)
        results = results[:limit]

        if (
            bm25_results and
            bm25_results[0].standout and
            bm25_results[0].id not in {result.id for result in results}
        ):
            results.append(bm25_results[0])

        return results

    def search_mmr(self, docs: list[SearchResult], landa: float=0.5, limit: int=10) -> list[SearchResult]:
        '''Re-rank and retrieved the sorted documents using MMR
            NOTE: Internal uses only
        '''
        documents = docs.copy()
        documents.sort(key= lambda result: result.hybrid_score, reverse=True)

        vectors = self.get_vectors(document.id for document in documents)

        S: list[SearchResult] = []

        while documents and len(S) < limit:
            best_mmr, best_i = -sys.float_info.max, 0

            for i, document in enumerate(documents):
                sims = [
                    math.similarity_cosine(
                        vectors[s.id],
                        vectors[document.id]
                    )
                    for s in S
                ]

                max_sim = max(sims) if sims else 0.0
                mmr = landa * document.hybrid_score - (1 - landa) * max_sim

                if (mmr > best_mmr):
                    best_mmr = mmr
                    best_i = i

            documents[best_i].mmr = best_mmr
            S.append(documents.pop(best_i))

        return S

    def search(self, query: str,
               k=60.0, landa=0.5,
               limit=5
    ) -> list[SearchResult]:
        hybrid_results = self.search_hybrid(
            query=query,
            limit=20, k=k
        )

        mmr_results = self.search_mmr(
            docs=hybrid_results,
            landa=landa,
            limit=limit
        )

        return [r for r in mmr_results if r.semantic_score >= self.SEMANTIC_THRESHOLD]

                