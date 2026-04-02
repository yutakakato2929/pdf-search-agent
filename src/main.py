#!/usr/bin/env python3
"""PDF Search Agent - 統合CLI

PDFドキュメントを活用したQ&A、および自律エージェント機能を提供します。
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Tokenizers の並列処理警告を抑制
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.chunker import Chunker
from src.knowledge.extractor import PDFExtractor
from src.knowledge.indexer import Indexer
from src.knowledge.retriever import HybridRetriever
from src.reasoning.qa import QAEngine

# 環境変数を読み込み
load_dotenv()

console = Console()


def setup_logging(verbose: bool = False):
    """ロギングを設定"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


def get_default_paths() -> dict[str, Path]:
    """デフォルトのパスを取得"""
    return {
        "pdfs": PROJECT_ROOT / "data" / "pdfs",
        "vectorstore": PROJECT_ROOT / "vectorstore",
        "bm25_index": PROJECT_ROOT / "bm25_index",
    }


def check_auth() -> bool:
    """認証情報の確認（OAuthトークンまたはAPIキー）"""
    oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if oauth_token:
        console.print("[dim]認証: OAuth token (サブスクリプション)[/dim]")
        return True
    elif api_key:
        console.print("[dim]認証: API key (従量課金)[/dim]")
        return True
    else:
        console.print("[red]エラー: 認証が設定されていません[/red]")
        console.print()
        console.print("[bold]【推奨】サブスクリプション（Pro/Max）を使用:[/bold]")
        console.print("  1. claude setup-token を実行してトークンを取得")
        console.print("  2. export CLAUDE_CODE_OAUTH_TOKEN=\"取得したトークン\"")
        console.print()
        console.print("[bold]【代替】APIキーを使用（従量課金）:[/bold]")
        console.print("  export ANTHROPIC_API_KEY='your-api-key'")
        return False


def check_index(paths: dict[str, Path]) -> bool:
    """インデックスの存在確認"""
    try:
        indexer = Indexer(paths["vectorstore"], paths["bm25_index"])
        collection = indexer.get_chroma_collection()
        if collection.count() == 0:
            console.print("[yellow]インデックスが空です。先に 'index' コマンドを実行してください。[/yellow]")
            return False
        return True
    except Exception:
        console.print("[yellow]インデックスが見つかりません。先に 'index' コマンドを実行してください。[/yellow]")
        return False


# =============================================================================
# メイン CLI グループ
# =============================================================================

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="詳細なログを出力")
@click.pass_context
def cli(ctx, verbose):
    """PDF Search Agent - 自律型AIエージェント

    PDFドキュメントを活用したQ&A、および自律エージェント機能を提供します。

    \b
    基本コマンド:
      index     - PDFドキュメントをインデックス化
      status    - インデックスの状態を表示
      qa        - Q&Aモード（従来版）

    \b
    自律エージェントコマンド:
      agent     - 自律エージェントを実行
      agent-qa  - 自律Q&Aモード
      serve     - MCPサーバーを起動
    """
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["paths"] = get_default_paths()


# =============================================================================
# 基本コマンド
# =============================================================================

@cli.command()
@click.option("--pdf-dir", type=click.Path(exists=True), help="PDFディレクトリ")
@click.pass_context
def index(ctx, pdf_dir):
    """PDFドキュメントをインデックス化"""
    paths = ctx.obj["paths"]
    pdf_path = Path(pdf_dir) if pdf_dir else paths["pdfs"]

    console.print(f"[bold]PDFディレクトリ:[/bold] {pdf_path}")

    # PDFファイルを列挙
    pdf_files = list(pdf_path.glob("*.pdf"))
    if not pdf_files:
        console.print("[yellow]PDFファイルが見つかりません[/yellow]")
        return

    console.print(f"[bold]検出されたPDF:[/bold] {len(pdf_files)}件")

    # PDF抽出
    extractor = PDFExtractor()
    chunker = Chunker()
    indexer = Indexer(paths["vectorstore"], paths["bm25_index"])

    all_chunks = []

    with console.status("[bold green]PDFを処理中..."):
        for pdf_file in pdf_files:
            console.print(f"  処理中: {pdf_file.name}")
            try:
                pages = extractor.extract(pdf_file)
                chunks = chunker.chunk_pages(pages)
                all_chunks.extend(chunks)
                console.print(f"    → {len(pages)}ページ, {len(chunks)}チャンク")
            except Exception as e:
                console.print(f"    [red]エラー: {e}[/red]")

    if not all_chunks:
        console.print("[yellow]チャンクが生成されませんでした[/yellow]")
        return

    # インデックス構築
    with console.status("[bold green]インデックスを構築中..."):
        indexer.index_chunks(all_chunks)

    console.print(f"[green]完了:[/green] {len(all_chunks)}チャンクをインデックス化しました")


@cli.command()
@click.pass_context
def status(ctx):
    """インデックスの状態を表示"""
    paths = ctx.obj["paths"]

    console.print(Panel("[bold]PDF Search Agent ステータス[/bold]", expand=False))
    console.print()

    # PDFディレクトリ
    pdf_path = paths["pdfs"]
    if pdf_path.exists():
        pdf_count = len(list(pdf_path.glob("*.pdf")))
        console.print(f"[bold]PDFディレクトリ:[/bold] {pdf_path}")
        console.print(f"  PDFファイル数: {pdf_count}")
    else:
        console.print(f"[bold]PDFディレクトリ:[/bold] {pdf_path} [yellow](未作成)[/yellow]")

    console.print()

    # ベクトルストア
    vectorstore_path = paths["vectorstore"]
    if vectorstore_path.exists():
        try:
            indexer = Indexer(paths["vectorstore"], paths["bm25_index"])
            collection = indexer.get_chroma_collection()
            chunk_count = collection.count()
            console.print(f"[bold]ベクトルストア:[/bold] {vectorstore_path}")
            console.print(f"  チャンク数: {chunk_count}")
        except Exception:
            console.print(f"[bold]ベクトルストア:[/bold] {vectorstore_path} [yellow](未初期化)[/yellow]")
    else:
        console.print(f"[bold]ベクトルストア:[/bold] {vectorstore_path} [yellow](未作成)[/yellow]")

    console.print()

    # BM25インデックス
    bm25_path = paths["bm25_index"]
    bm25_file = bm25_path / "bm25_index.pkl"
    if bm25_file.exists():
        console.print(f"[bold]BM25インデックス:[/bold] {bm25_file}")
    else:
        console.print(f"[bold]BM25インデックス:[/bold] [yellow](未作成)[/yellow]")

    console.print()

    # Agent SDK
    try:
        from src.agent import AGENT_SDK_AVAILABLE, check_auth as agent_check_auth
        if AGENT_SDK_AVAILABLE:
            console.print("[bold]Claude Agent SDK:[/bold] [green]利用可能[/green]")
        else:
            console.print("[bold]Claude Agent SDK:[/bold] [yellow]未インストール[/yellow]")

        # 認証状態
        auth_ok, auth_method = agent_check_auth()
        if auth_ok:
            console.print(f"[bold]認証:[/bold] [green]{auth_method}[/green]")
        else:
            console.print("[bold]認証:[/bold] [yellow]未設定[/yellow]")
    except ImportError:
        console.print("[bold]Claude Agent SDK:[/bold] [yellow]未インストール[/yellow]")


@cli.command()
@click.option("--question", "-q", help="質問（指定しない場合は対話モード）")
@click.pass_context
def qa(ctx, question):
    """Q&Aモード: ドキュメントを参照して質問に回答（従来版）"""
    if not check_auth():
        return

    paths = ctx.obj["paths"]
    if not check_index(paths):
        return

    indexer = Indexer(paths["vectorstore"], paths["bm25_index"])
    retriever = HybridRetriever(indexer)
    qa_engine = QAEngine(retriever)

    if question:
        with console.status("[bold green]回答を生成中..."):
            answer = qa_engine.answer(question)
        console.print()
        console.print(answer)
    else:
        qa_engine.interactive_mode()


# =============================================================================
# 自律エージェントコマンド
# =============================================================================

@cli.command("agent")
@click.argument("goal")
@click.option("--doc-type", "-t", default="ドキュメント", help="ドキュメントの種類")
@click.pass_context
def agent_run(ctx, goal, doc_type):
    """自律エージェントを実行

    GOAL: 達成したいゴールを自然言語で指定
    """
    if not check_auth():
        return

    try:
        from src.agent import run_agent, print_message, AGENT_SDK_AVAILABLE
    except ImportError:
        console.print("[red]エラー: エージェントモジュールをインポートできません[/red]")
        return

    if not AGENT_SDK_AVAILABLE:
        console.print("[red]エラー: Claude Agent SDK がインストールされていません[/red]")
        console.print("pip install claude-agent-sdk を実行してください")
        return

    console.print(Panel(f"[bold]Goal:[/bold] {goal}", title="PDF Search Agent", expand=False))
    console.print()

    async def main():
        async for message in run_agent(goal, document_type=doc_type):
            print_message(message)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]中断されました[/yellow]")
    except Exception as e:
        console.print(f"[red]エラー: {e}[/red]")


@cli.command("agent-qa")
@click.argument("question")
@click.option("--doc-type", "-t", default="ドキュメント", help="ドキュメントの種類")
@click.pass_context
def agent_qa(ctx, question, doc_type):
    """自律Q&Aモード: エージェントが自律的にドキュメントを検索して回答

    QUESTION: 質問を自然言語で指定
    """
    if not check_auth():
        return

    try:
        from src.agent import autonomous_qa, print_message, AGENT_SDK_AVAILABLE
    except ImportError:
        console.print("[red]エラー: エージェントモジュールをインポートできません[/red]")
        return

    if not AGENT_SDK_AVAILABLE:
        console.print("[red]エラー: Claude Agent SDK がインストールされていません[/red]")
        return

    console.print(Panel(f"[bold]Question:[/bold] {question}", title="自律Q&Aモード", expand=False))
    console.print()

    async def main():
        async for message in autonomous_qa(question, doc_type):
            print_message(message)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]中断されました[/yellow]")
    except Exception as e:
        console.print(f"[red]エラー: {e}[/red]")


@cli.command()
def serve():
    """MCPサーバーを起動

    Claude Code や他のMCPクライアントから接続できる
    ナレッジサーバーを起動します。
    """
    console.print(Panel("[bold]PDF Search MCP Server[/bold]", expand=False))
    console.print()
    console.print("MCPサーバーを起動中... (Ctrl+Cで終了)")
    console.print()

    try:
        from src.mcp_server import main as mcp_main
        mcp_main()
    except ImportError:
        console.print("[red]エラー: MCPサーバーモジュールをインポートできません[/red]")
    except KeyboardInterrupt:
        console.print("\n[yellow]サーバーを終了しました[/yellow]")
    except Exception as e:
        console.print(f"[red]エラー: {e}[/red]")


# =============================================================================
# エントリーポイント
# =============================================================================

def main():
    """メインエントリーポイント"""
    cli()


if __name__ == "__main__":
    main()
