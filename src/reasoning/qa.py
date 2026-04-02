"""Q&Aモードモジュール

PDFドキュメントを参照して質問に回答する。
"""

import logging
import os
from pathlib import Path
from typing import Optional

from anthropic import Anthropic

from ..knowledge.retriever import HybridRetriever, RetrievalResult

logger = logging.getLogger(__name__)


def get_anthropic_client() -> Anthropic:
    """認証情報を自動検出してAnthropicクライアントを作成"""
    oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if oauth_token:
        return Anthropic(auth_token=oauth_token)
    elif api_key:
        return Anthropic(api_key=api_key)
    else:
        # デフォルト（環境変数から自動検出を試みる）
        return Anthropic()


class CitationManager:
    """出典管理"""

    def __init__(self):
        self.citations: list[dict] = []

    def clear(self):
        """出典をクリア"""
        self.citations = []

    def add_chunk(self, chunk) -> int:
        """チャンクを出典として追加"""
        idx = len(self.citations) + 1
        self.citations.append({
            "index": idx,
            "file": chunk.file,
            "version": chunk.version,
            "section": chunk.section,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
        })
        return idx

    def format_with_warning(self) -> str:
        """出典をフォーマットして返す"""
        if not self.citations:
            return ""

        lines = ["[出典]"]
        for c in self.citations:
            page_info = f"p.{c['page_start']}"
            if c['page_end'] != c['page_start']:
                page_info = f"p.{c['page_start']}-{c['page_end']}"

            lines.append(
                f"  [{c['index']}] {c['file']} ({c['version']}) "
                f"{c['section']} {page_info}"
            )

        return "\n".join(lines)


class QAEngine:
    """Q&A応答エンジン"""

    def __init__(
        self,
        retriever: HybridRetriever,
        model: str = "claude-sonnet-4-20250514",
        prompts_dir: Optional[Path] = None,
    ):
        self.retriever = retriever
        self.model = model
        self.client = get_anthropic_client()
        self.citation_manager = CitationManager()

        # システムプロンプトを読み込み
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent / "prompts"
        self.system_prompt = self._load_system_prompt(prompts_dir / "qa_system.txt")

    def _load_system_prompt(self, path: Path) -> str:
        """システムプロンプトを読み込む"""
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning(f"System prompt not found: {path}")
            return ""

    def answer(self, question: str, top_k: int = 5) -> str:
        """質問に回答する"""
        self.citation_manager.clear()

        # 関連ドキュメントを検索
        results = self.retriever.retrieve(question, top_k=top_k)

        if not results:
            return self._format_no_results_response()

        # コンテキストを構築
        context = self._build_context(results)

        # LLMで回答を生成
        response = self._generate_response(question, context)

        # 出典を追加
        citations = self.citation_manager.format_with_warning()
        if citations:
            response = f"{response}\n\n{citations}"

        return response

    def _build_context(self, results: list[RetrievalResult]) -> str:
        """検索結果からコンテキストを構築"""
        context_parts = []

        for result in results:
            chunk = result.chunk
            citation_idx = self.citation_manager.add_chunk(chunk)

            context_parts.append(
                f"[参考資料 {citation_idx}]\n"
                f"ファイル: {chunk.file} ({chunk.version})\n"
                f"セクション: {chunk.section}\n"
                f"ページ: {chunk.page_start}-{chunk.page_end}\n"
                f"内容:\n{chunk.text}\n"
            )

        return "\n---\n".join(context_parts)

    def _generate_response(self, question: str, context: str) -> str:
        """LLMで回答を生成"""
        user_message = f"""以下の参考資料に基づいて、質問に回答してください。

## 参考資料

{context}

## 質問

{question}
"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ],
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def _format_no_results_response(self) -> str:
        """検索結果がない場合の回答"""
        return (
            "[回答]\n"
            "申し訳ありませんが、お探しの情報はドキュメントに記載が見当たりませんでした。\n"
            "質問の内容を変えて再度お試しいただくか、直接ドキュメントをご確認ください。\n\n"
            "[出典]\n"
            "該当なし"
        )

    def interactive_mode(self):
        """対話モード"""
        print("Q&A モードを開始します。")
        print("質問を入力してください（終了するには 'exit' または 'quit' と入力）。")
        print("-" * 50)

        while True:
            try:
                question = input("\n質問: ").strip()

                if not question:
                    continue

                if question.lower() in ("exit", "quit", "q"):
                    print("Q&Aモードを終了します。")
                    break

                print("\n回答を生成中...")
                answer = self.answer(question)
                print(f"\n{answer}")

            except KeyboardInterrupt:
                print("\n\nQ&Aモードを終了します。")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"エラーが発生しました: {e}")
