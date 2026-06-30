from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


# Encoder dùng chung toàn process (tránh load model 2 lần → tiết kiệm RAM).
_SHARED_ENCODER = None


def get_shared_encoder():
    """Trả về SentenceTransformer dùng chung (lazy-load 1 lần)."""
    global _SHARED_ENCODER
    if _SHARED_ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _SHARED_ENCODER = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _SHARED_ENCODER


def warmup_embedding():
    """Khởi tạo đầy đủ stack torch/transformers TRƯỚC khi nạp pypdf/underthesea.

    Trên Windows, nếu native runtime của pypdf hoặc underthesea_core (Rust) được
    nạp trước torch, việc load/encode model embedding sẽ crash 0xC0000005
    (access violation). Gọi hàm này ở đầu mỗi entrypoint có dùng pipeline.
    """
    enc = get_shared_encoder()
    enc.encode(["warmup"], show_progress_bar=False)
    return enc


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    from underthesea import word_tokenize

    segmented = word_tokenize(text, format="text")
    return segmented.replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = chunks
        self.corpus_tokens = [
            segment_vietnamese(chunk["text"]).split()
            for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []

        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for i in top_indices:
            if scores[i] > 0:
                results.append(SearchResult(
                    text=self.documents[i]["text"],
                    score=float(scores[i]),
                    metadata=self.documents[i].get("metadata", {}),
                    method="bm25",
                ))
        return results


class DenseSearch:
    def __init__(self):
        self.client = None
        self._encoder = None

    def _get_client(self):
        if self.client is None:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)
        return self.client

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = get_shared_encoder()
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        import gc
        from qdrant_client.models import Distance, VectorParams, PointStruct

        encoder = self._get_encoder()
        gc.collect()
        client = self._get_client()
        client.recreate_collection(
            collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        for start, c in enumerate(chunks):
            vector = encoder.encode([c["text"]], batch_size=1, show_progress_bar=False)[0]
            client.upsert(
                collection,
                [
                    PointStruct(
                        id=start,
                        vector=vector.tolist(),
                        payload={**c.get("metadata", {}), "text": c["text"]},
                    )
                ],
            )
            if (start + 1) % 20 == 0:
                gc.collect()

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        query_vector = self._get_encoder().encode(query).tolist()
        response = self._get_client().query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
        )
        return [
            SearchResult(
                text=pt.payload["text"],
                score=float(pt.score),
                metadata=pt.payload,
                method="dense",
            )
            for pt in response.points
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[str, dict] = {}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    sorted_items = sorted(
        rrf_scores.values(), key=lambda x: x["score"], reverse=True
    )[:top_k]

    return [
        SearchResult(
            text=item["result"].text,
            score=item["score"],
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in sorted_items
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense: DenseSearch | None = None

    def _get_dense(self) -> DenseSearch:
        if self.dense is None:
            self.dense = DenseSearch()
        return self.dense

    def index(self, chunks: list[dict]) -> None:
        import gc
        # Đảm bảo encoder torch đã sẵn sàng trước khi dùng underthesea (BM25),
        # phòng trường hợp entrypoint chưa gọi warmup_embedding().
        self._get_dense()._get_encoder()
        self.bm25.index(chunks)
        gc.collect()
        self._get_dense().index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self._get_dense().search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
