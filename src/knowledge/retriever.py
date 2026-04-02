"""ハイブリッド検索モジュール

ベクトル検索（ChromaDB）とキーワード検索（BM25）を組み合わせ、
Reciprocal Rank Fusion（RRF）でスコア統合後、CrossEncoderで再ランキング。
"""

import logging
from dataclasses import dataclass

import numpy as np
from sentence_transformers import CrossEncoder

from .chunker import Chunk
from .indexer import Indexer

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """検索結果"""
    chunk: Chunk
    score: float
    source: str  # "vector", "bm25", "hybrid"


class HybridRetriever:
    """ハイブリッド検索 + 再ランキング"""

    # RRFのパラメータ
    RRF_K = 60

    def __init__(
        self,
        indexer: Indexer,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.indexer = indexer
        self.reranker = CrossEncoder(reranker_model)

        # E5モデルのクエリプレフィックス
        self._use_query_prefix = "e5" in indexer.embedding_fn._model_name.lower()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        vector_top_k: int = 10,
        bm25_top_k: int = 10,
        rerank_top_k: int = 20,
    ) -> list[RetrievalResult]:
        """ハイブリッド検索を実行

        1. ベクトル検索（ChromaDB）で top-k 取得
        2. BM25検索で top-k 取得
        3. RRFでスコア統合
        4. 上位をCrossEncoderで再ランキング
        5. 最終的な top-k を返却
        """
        # ベクトル検索
        vector_results = self._vector_search(query, vector_top_k)

        # BM25検索
        bm25_results = self._bm25_search(query, bm25_top_k)

        # RRFでスコア統合
        fused_results = self._reciprocal_rank_fusion(
            vector_results, bm25_results, top_k=rerank_top_k
        )

        if not fused_results:
            return []

        # CrossEncoderで再ランキング
        reranked_results = self._rerank(query, fused_results, top_k)

        return reranked_results

    def _vector_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """ベクトル検索"""
        try:
            collection = self.indexer.get_chroma_collection()

            # E5モデルの場合、クエリにプレフィックスを付ける
            search_query = query
            if self._use_query_prefix:
                search_query = f"query: {query}"

            results = collection.query(
                query_texts=[search_query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            retrieval_results = []
            if results and results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    document = results["documents"][0][i]
                    distance = results["distances"][0][i]

                    # cosine distanceをスコアに変換（1 - distance）
                    score = 1.0 - distance

                    chunk = Chunk(
                        chunk_id=chunk_id,
                        chunk_type=metadata.get("chunk_type", "fixed"),
                        file=metadata.get("file", ""),
                        version=metadata.get("version", ""),
                        page_start=metadata.get("page_start", 0),
                        page_end=metadata.get("page_end", 0),
                        section=metadata.get("section", ""),
                        section_path=metadata.get("section_path", ""),
                        text=document,
                    )

                    retrieval_results.append(RetrievalResult(
                        chunk=chunk,
                        score=score,
                        source="vector",
                    ))

            return retrieval_results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """BM25検索"""
        bm25, chunks = self.indexer.get_bm25()

        if bm25 is None or not chunks:
            logger.warning("BM25 index not available")
            return []

        # クエリをトークン化
        query_tokens = self.indexer._tokenize(query)

        # BM25スコアを計算
        scores = bm25.get_scores(query_tokens)

        # 上位k件を取得
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(RetrievalResult(
                    chunk=chunks[idx],
                    score=scores[idx],
                    source="bm25",
                ))

        return results

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[RetrievalResult],
        bm25_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Reciprocal Rank Fusion（RRF）でスコア統合"""
        chunk_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        # ベクトル検索結果のRRFスコア
        for rank, result in enumerate(vector_results, 1):
            chunk_id = result.chunk.chunk_id
            rrf_score = 1.0 / (self.RRF_K + rank)
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score
            chunk_map[chunk_id] = result.chunk

        # BM25検索結果のRRFスコア
        for rank, result in enumerate(bm25_results, 1):
            chunk_id = result.chunk.chunk_id
            rrf_score = 1.0 / (self.RRF_K + rank)
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score
            chunk_map[chunk_id] = result.chunk

        # スコアでソート
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return [
            RetrievalResult(
                chunk=chunk_map[chunk_id],
                score=score,
                source="hybrid",
            )
            for chunk_id, score in sorted_chunks
        ]

    def _rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """CrossEncoderで再ランキング"""
        if not results:
            return []

        # クエリとドキュメントのペアを作成
        pairs = [(query, result.chunk.text) for result in results]

        # スコアを計算
        scores = self.reranker.predict(pairs)

        # スコアでソート
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # 上位k件を返却
        return [
            RetrievalResult(
                chunk=result.chunk,
                score=float(score),
                source="reranked",
            )
            for result, score in scored_results[:top_k]
        ]

    def vector_only_search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """ベクトル検索のみ（デバッグ用）"""
        return self._vector_search(query, top_k)

    def bm25_only_search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """BM25検索のみ（デバッグ用）"""
        return self._bm25_search(query, top_k)
