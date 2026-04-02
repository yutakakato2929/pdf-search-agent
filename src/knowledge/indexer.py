"""インデックス構築モジュール

ChromaDB（ベクトル検索）とBM25（キーワード検索）の両方にインデックスを構築する。
"""

import logging
import pickle
import re
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .chunker import Chunk

logger = logging.getLogger(__name__)


class EmbeddingFunction:
    """sentence-transformersを使ったEmbedding関数

    将来的にVoyage AIやOpenAIに切り替え可能な設計。
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.model = SentenceTransformer(model_name)
        self._model_name = model_name

    def __call__(self, input: list[str]) -> list[list[float]]:
        # E5モデルの場合、ドキュメントには "passage: " プレフィックスを付ける
        if "e5" in self._model_name.lower():
            input = [f"passage: {text}" for text in input]
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()


class ChromaEmbeddingFunction:
    """ChromaDB用のEmbedding関数ラッパー"""

    def __init__(self, embedding_fn: EmbeddingFunction):
        self._embedding_fn = embedding_fn

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embedding_fn(input)


class Indexer:
    """ベクトルDB + BM25インデックス構築"""

    DEFAULT_COLLECTION_NAME = "pdf_docs"

    def __init__(
        self,
        vectorstore_path: Path,
        bm25_index_path: Path,
        embedding_model: str = "intfloat/multilingual-e5-base",
        collection_name: str | None = None,
    ):
        self.vectorstore_path = Path(vectorstore_path)
        self.bm25_index_path = Path(bm25_index_path)
        self.embedding_fn = EmbeddingFunction(embedding_model)
        self.collection_name = collection_name or self.DEFAULT_COLLECTION_NAME

        # ChromaDB クライアント初期化
        self.vectorstore_path.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.vectorstore_path),
            settings=Settings(anonymized_telemetry=False),
        )

        # BM25インデックス用
        self.bm25_index_path.mkdir(parents=True, exist_ok=True)
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_chunks: list[Chunk] = []

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """チャンクをインデックスに登録"""
        if not chunks:
            logger.warning("No chunks to index")
            return

        logger.info(f"Indexing {len(chunks)} chunks...")

        # ChromaDBにインデックス
        self._index_to_chromadb(chunks)

        # BM25インデックス構築
        self._index_to_bm25(chunks)

        logger.info("Indexing completed")

    def _index_to_chromadb(self, chunks: list[Chunk]) -> None:
        """ChromaDBにチャンクを登録"""
        # 既存のコレクションがあれば削除して再作成
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass

        collection = self.chroma_client.create_collection(
            name=self.collection_name,
            embedding_function=ChromaEmbeddingFunction(self.embedding_fn),
            metadata={"hnsw:space": "cosine"},
        )

        # バッチでインデックス（ChromaDBの制限に合わせて分割）
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            ids = [chunk.chunk_id for chunk in batch]
            documents = [chunk.text for chunk in batch]
            metadatas = [
                {
                    "chunk_type": chunk.chunk_type,
                    "file": chunk.file,
                    "version": chunk.version,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section": chunk.section,
                    "section_path": chunk.section_path,
                }
                for chunk in batch
            ]

            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

        logger.info(f"Indexed {len(chunks)} chunks to ChromaDB")

    def _index_to_bm25(self, chunks: list[Chunk]) -> None:
        """BM25インデックスを構築"""
        # テキストをトークン化（簡易的な日本語対応）
        tokenized_corpus = [self._tokenize(chunk.text) for chunk in chunks]

        self._bm25 = BM25Okapi(tokenized_corpus)
        self._bm25_chunks = chunks

        # インデックスを保存
        index_file = self.bm25_index_path / "bm25_index.pkl"
        with open(index_file, "wb") as f:
            pickle.dump({
                "bm25": self._bm25,
                "chunks": self._bm25_chunks,
            }, f)

        logger.info(f"BM25 index saved to {index_file}")

    def _tokenize(self, text: str) -> list[str]:
        """テキストをトークン化（簡易実装）

        日本語の場合、文字単位の分割も含める。
        より高度な実装ではMeCabなどを使用。
        """
        # 空白・句読点で分割
        tokens = re.split(r'[\s、。，．・\n]+', text)

        # 空のトークンを除去
        tokens = [t for t in tokens if t.strip()]

        return tokens

    def load_bm25_index(self) -> bool:
        """保存済みのBM25インデックスを読み込む"""
        index_file = self.bm25_index_path / "bm25_index.pkl"
        if not index_file.exists():
            return False

        try:
            with open(index_file, "rb") as f:
                data = pickle.load(f)
                self._bm25 = data["bm25"]
                self._bm25_chunks = data["chunks"]
            logger.info("BM25 index loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    def get_chroma_collection(self):
        """ChromaDBコレクションを取得"""
        return self.chroma_client.get_collection(
            name=self.collection_name,
            embedding_function=ChromaEmbeddingFunction(self.embedding_fn),
        )

    def get_bm25(self) -> tuple[Optional[BM25Okapi], list[Chunk]]:
        """BM25インデックスとチャンクリストを取得"""
        if self._bm25 is None:
            self.load_bm25_index()
        return self._bm25, self._bm25_chunks
