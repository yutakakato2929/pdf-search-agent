#!/usr/bin/env python3
"""PDF Search Agent - Claude Agent SDK ベースの自律型エージェント

高自律性を持つAIエージェント。ゴールを与えると、計画→実行→検証→修正を
自律的に繰り返してタスクを完了する。

認証方法:
- CLAUDE_CODE_OAUTH_TOKEN: claude.ai サブスクリプション（Pro/Max）で利用
- ANTHROPIC_API_KEY: API キーで利用（従量課金）
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import AsyncIterator, Any

# Tokenizers の並列処理警告を抑制
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent

# Claude Agent SDK のインポート
try:
    from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition
    from claude_agent_sdk import AssistantMessage, ResultMessage, SystemMessage
    AGENT_SDK_AVAILABLE = True
except ImportError:
    logger.warning("claude-agent-sdk not installed. Agent features will be limited.")
    AGENT_SDK_AVAILABLE = False


def check_auth() -> tuple[bool, str]:
    """認証情報を確認

    Returns:
        (認証OK, 認証方式の説明)
    """
    oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if oauth_token:
        return True, "OAuth token (サブスクリプション)"
    elif api_key:
        return True, "API key (従量課金)"
    else:
        return False, "未設定"


def get_auth_help() -> str:
    """認証設定のヘルプを取得"""
    return """
認証が設定されていません。以下のいずれかを設定してください:

【推奨】サブスクリプション（Pro/Max）を使用:
  1. claude setup-token を実行してトークンを取得
  2. export CLAUDE_CODE_OAUTH_TOKEN="取得したトークン"

【代替】APIキーを使用（従量課金）:
  1. https://console.anthropic.com/ でAPIキーを取得
  2. export ANTHROPIC_API_KEY="your-api-key"
""".strip()


def get_system_prompt(document_type: str = "PDFドキュメント") -> str:
    """システムプロンプトを生成

    Args:
        document_type: ドキュメントの種類（例: "技術マニュアル", "製品仕様書"）
    """
    return f"""あなたは{document_type}の専門家エージェントです。

## 能力
- ドキュメントを検索して正確な情報を提供（search_docs）
- 質問に対して参考資料を引用しながら回答

## 行動原則
1. まず要件を明確に理解する
2. 必要な情報をドキュメントから検索する
3. 検索結果に基づいて回答を生成する
4. 出典を明示する

## 出力形式
- 回答には必ず出典（ファイル名、セクション、ページ）を含める
- 具体例がある場合は提示する
- 検索結果に該当がない場合は「記載なし」と明示

## 注意事項
- ドキュメントに記載がない場合は推測しない
- 不明点があれば確認を求める
"""


# サブエージェント定義
SUBAGENTS = {
    "qa-expert": {
        "description": "ドキュメントに基づくQ&A専門家。質問に対して参考資料を引用しながら回答。",
        "prompt": """
## 役割
ドキュメントに関する質問に、検索結果を根拠として回答します。

## 回答形式
1. [回答] 質問への直接的な回答
2. [参考] 関連する追加情報（ある場合）
3. [出典] ドキュメントの該当箇所（ファイル名、セクション、ページ）

## 注意事項
- ドキュメントに記載がない場合は「記載なし」と明示
- 推測で回答せず、必ず検索結果に基づく
""",
        "tools": ["mcp__pdf_search__search_docs"],
    },
}


def get_mcp_server_config() -> dict[str, Any]:
    """MCPサーバー設定を取得"""
    python_path = PROJECT_ROOT / ".venv" / "bin" / "python"
    mcp_server_path = PROJECT_ROOT / "src" / "mcp_server.py"

    return {
        "pdf_search": {
            "command": str(python_path),
            "args": [str(mcp_server_path)],
        }
    }


def get_agent_definitions() -> dict[str, Any]:
    """サブエージェント定義を取得（Claude Agent SDK形式）"""
    if not AGENT_SDK_AVAILABLE:
        return {}

    return {
        name: AgentDefinition(
            description=config["description"],
            prompt=config["prompt"],
            tools=config["tools"],
        )
        for name, config in SUBAGENTS.items()
    }


async def run_agent(
    goal: str,
    allowed_tools: list[str] | None = None,
    use_subagents: bool = True,
    permission_mode: str = "acceptEdits",
    document_type: str = "PDFドキュメント",
) -> AsyncIterator[Any]:
    """メインエージェントを実行

    Args:
        goal: 達成するゴール
        allowed_tools: 許可するツールのリスト
        use_subagents: サブエージェントを使用するか
        permission_mode: パーミッションモード
        document_type: ドキュメントの種類

    Yields:
        エージェントからのメッセージ
    """
    if not AGENT_SDK_AVAILABLE:
        raise RuntimeError("claude-agent-sdk is not installed")

    # デフォルトのツール
    if allowed_tools is None:
        allowed_tools = [
            # 基本ツール
            "Read", "Write", "Glob", "Grep",
            # MCPツール（pdf_search）
            "mcp__pdf_search__search_docs",
            "mcp__pdf_search__get_index_status",
        ]
        if use_subagents:
            allowed_tools.append("Agent")

    # オプション構築
    options = ClaudeAgentOptions(
        system_prompt=get_system_prompt(document_type),
        allowed_tools=allowed_tools,
        permission_mode=permission_mode,
        mcp_servers=get_mcp_server_config(),
    )

    # サブエージェントを追加
    if use_subagents:
        options.agents = get_agent_definitions()

    # エージェントを実行
    async for message in query(prompt=goal, options=options):
        yield message


async def autonomous_qa(question: str, document_type: str = "ドキュメント") -> AsyncIterator[Any]:
    """自律的なQ&A

    Args:
        question: 質問
        document_type: ドキュメントの種類

    Yields:
        エージェントからのメッセージ
    """
    goal = f"""以下の質問に、{document_type}を検索して回答してください。

## 質問
{question}

## 回答形式
1. [回答] 質問への直接的な回答
2. [参考] 関連する追加情報（ある場合）
3. [出典] ドキュメントの該当箇所（ファイル名、セクション、ページ）

ドキュメントに記載がない場合は「記載なし」と明示してください。
"""
    async for message in run_agent(goal, document_type=document_type):
        yield message


def print_message(message: Any) -> None:
    """メッセージを表示"""
    if not AGENT_SDK_AVAILABLE:
        print(str(message))
        return

    if isinstance(message, AssistantMessage):
        for block in message.content:
            if hasattr(block, "text"):
                print(block.text)
            elif hasattr(block, "name"):
                print(f"[Tool: {block.name}]")
    elif isinstance(message, ResultMessage):
        print(f"\n[完了: {message.subtype}]")
    elif isinstance(message, SystemMessage):
        if hasattr(message, "subtype") and message.subtype == "init":
            print("[エージェント初期化完了]")


async def run_interactive(goal: str, document_type: str = "ドキュメント") -> None:
    """インタラクティブにエージェントを実行"""
    print(f"\n[Goal] {goal}\n")
    print("-" * 60)

    async for message in run_agent(goal, document_type=document_type):
        print_message(message)

    print("-" * 60)


# CLI エントリーポイント
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF Search Agent")
    parser.add_argument("goal", nargs="?", help="達成するゴール")
    parser.add_argument("--qa", "-q", help="Q&Aモード（質問を指定）")
    parser.add_argument("--doc-type", "-t", default="ドキュメント", help="ドキュメントの種類")

    args = parser.parse_args()

    if args.qa:
        async def main():
            async for msg in autonomous_qa(args.qa, args.doc_type):
                print_message(msg)
        asyncio.run(main())
    elif args.goal:
        asyncio.run(run_interactive(args.goal, args.doc_type))
    else:
        parser.print_help()
