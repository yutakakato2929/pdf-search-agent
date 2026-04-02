#!/usr/bin/env python3
"""PDF Search MCP Server

PDFドキュメントをナレッジベースとして
MCPツールを提供するサーバー。
"""

import logging
import os
import sys
from pathlib import Path

# Tokenizers の並列処理警告を抑制
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.indexer import Indexer
from src.knowledge.retriever import HybridRetriever, RetrievalResult

# 環境変数を読み込み
load_dotenv()

# ロギング設定（stderrに出力）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# FastMCPサーバーを初期化
mcp = FastMCP("pdf-search")

# デフォルトパス
DEFAULT_PATHS = {
    "vectorstore": PROJECT_ROOT / "vectorstore",
    "bm25_index": PROJECT_ROOT / "bm25_index",
}


def get_retriever() -> HybridRetriever | None:
    """Retrieverを取得（シングルトン的に遅延初期化）"""
    if not hasattr(get_retriever, "_instance"):
        try:
            indexer = Indexer(
                DEFAULT_PATHS["vectorstore"],
                DEFAULT_PATHS["bm25_index"],
            )
            # インデックスの存在確認
            collection = indexer.get_chroma_collection()
            if collection.count() == 0:
                logger.warning("Index is empty. Run 'index' command first.")
                get_retriever._instance = None
            else:
                get_retriever._instance = HybridRetriever(indexer)
                logger.info("HybridRetriever initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            get_retriever._instance = None
    return get_retriever._instance


def format_search_results(results: list[RetrievalResult]) -> str:
    """検索結果をフォーマット"""
    if not results:
        return "検索結果が見つかりませんでした。"

    formatted = []
    for i, result in enumerate(results, 1):
        chunk = result.chunk
        formatted.append(f"""
[{i}] スコア: {result.score:.3f}
ファイル: {chunk.file}
セクション: {chunk.section}
ページ: {chunk.page_start}-{chunk.page_end}

{chunk.text[:500]}{"..." if len(chunk.text) > 500 else ""}
""")
    return "\n---\n".join(formatted)


@mcp.tool()
def search_docs(query: str, top_k: int = 5) -> str:
    """ドキュメントを検索して関連情報を取得します。

    Args:
        query: 検索クエリ
        top_k: 取得する結果の最大数（デフォルト: 5）

    Returns:
        検索結果（関連するドキュメントの抜粋、出典情報付き）
    """
    retriever = get_retriever()
    if retriever is None:
        return "エラー: インデックスが初期化されていません。先に 'index' コマンドを実行してください。"

    try:
        results = retriever.retrieve(query, top_k=top_k)
        return format_search_results(results)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"検索エラー: {e}"


@mcp.tool()
def get_index_status() -> str:
    """インデックスの状態を取得します。

    Returns:
        インデックスの状態情報（チャンク数、ファイル数など）
    """
    output = []
    output.append("## PDF Search インデックス状態")
    output.append("")

    # ベクトルストア
    vectorstore_path = DEFAULT_PATHS["vectorstore"]
    if vectorstore_path.exists():
        try:
            indexer = Indexer(
                DEFAULT_PATHS["vectorstore"],
                DEFAULT_PATHS["bm25_index"],
            )
            collection = indexer.get_chroma_collection()
            chunk_count = collection.count()
            output.append(f"**ベクトルストア:** {vectorstore_path}")
            output.append(f"  - チャンク数: {chunk_count}")
        except Exception as e:
            output.append(f"**ベクトルストア:** エラー ({e})")
    else:
        output.append("**ベクトルストア:** 未作成")

    output.append("")

    # BM25インデックス
    bm25_path = DEFAULT_PATHS["bm25_index"]
    bm25_file = bm25_path / "bm25_index.pkl"
    if bm25_file.exists():
        output.append(f"**BM25インデックス:** {bm25_file}")
    else:
        output.append("**BM25インデックス:** 未作成")

    return "\n".join(output)


def main():
    """MCPサーバーを起動"""
    logger.info("Starting PDF Search MCP Server...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
