# PDF Search Agent

PDFドキュメントをナレッジベースとして活用する**自律型AIエージェント**。

Claude Agent SDK を使用し、ゴールを与えると自律的に計画→実行→検証を繰り返してタスクを完了します。

## 機能

### 自律エージェント機能
- **自律Q&A**: ドキュメントを自律的に検索して質問に回答
- **MCPサーバー**: Claude Code や他のMCPクライアントから利用可能

### 従来機能
- **Q&Aモード**: PDFドキュメントを参照して質問に回答

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. 認証の設定

プロジェクトルートに `.env` ファイルを作成して認証情報を設定します。

#### 【推奨】サブスクリプション（Pro/Max）を使用

claude.ai の Pro または Max プランで利用できます。API の従量課金は発生しません。

```bash
# トークンを取得
claude setup-token

# .env ファイルを作成
echo 'CLAUDE_CODE_OAUTH_TOKEN="取得したトークン"' > .env
```

#### 【代替】APIキーを使用（従量課金）

```bash
echo 'ANTHROPIC_API_KEY="your-api-key"' > .env
```

> **Note**: `.env` ファイルは `.gitignore` に含まれているため、リポジトリにコミットされません。

### 3. PDFドキュメントの配置

```
data/pdfs/
├── Document1.pdf
├── Document2.pdf
└── ...
```

### 4. インデックスの構築

```bash
python src/main.py index
```

## 使い方

### 基本コマンド

```bash
# ステータス確認
python src/main.py status

# インデックス構築
python src/main.py index --pdf-dir ./data/pdfs/
```

### 自律エージェントモード（推奨）

```bash
# 自律Q&A
python src/main.py agent-qa "この製品の特徴は何ですか？"

# ドキュメントタイプを指定
python src/main.py agent-qa "設定方法を教えて" --doc-type "技術マニュアル"

# 汎用エージェント（自然言語でゴール指定）
python src/main.py agent "APIの使い方をまとめて"
```

### 従来モード

```bash
# Q&A（対話モード）
python src/main.py qa

# Q&A（単発）
python src/main.py qa -q "この機能の使い方は？"
```

### MCPサーバーとして利用（Claude Code 連携）

Claude Code から直接ドキュメントを検索・活用できます。

#### 1. MCPサーバーの登録

```bash
# Claude Code に MCPサーバーを登録
claude mcp add pdf_search -s user -- \
  /path/to/pdf-search-agent/.venv/bin/python \
  /path/to/pdf-search-agent/src/mcp_server.py
```

※ `/path/to/pdf-search-agent` は実際のパスに置き換えてください。

#### 2. Claude Code を再起動

MCPサーバーの設定を反映するため、Claude Code を再起動します。

#### 3. 登録確認

Claude Code で `/mcp` コマンドを実行して、`pdf_search` が表示されることを確認します。

以下のツールが利用可能になります:
- `mcp__pdf_search__search_docs` - ドキュメント検索
- `mcp__pdf_search__get_index_status` - インデックス状態確認

#### 4. 使用例

Claude Code で自然言語で質問するだけで、ドキュメントを検索して回答します:

```
この製品のインストール手順を教えて
```

#### 手動設定（代替）

`~/.claude.json` に追加:

```json
{
  "mcpServers": {
    "pdf_search": {
      "command": "/path/to/pdf-search-agent/.venv/bin/python",
      "args": ["/path/to/pdf-search-agent/src/mcp_server.py"]
    }
  }
}
```

### MCPサーバーをスタンドアロンで起動

他のMCPクライアントから利用する場合:

```bash
python src/main.py serve
```

## ディレクトリ構成

```
pdf-search-agent/
├── data/pdfs/           # PDFドキュメント格納場所
├── vectorstore/         # ChromaDB 永続化データ
├── bm25_index/          # BM25 インデックス
├── src/
│   ├── main.py          # CLI エントリーポイント
│   ├── agent.py         # 自律エージェント（Claude Agent SDK）
│   ├── mcp_server.py    # MCPサーバー
│   ├── knowledge/       # Knowledge Layer（RAG）
│   ├── reasoning/       # Reasoning Layer
│   └── prompts/         # システムプロンプト
├── requirements.txt
└── README.md
```

## 技術スタック

- **LLM**: Claude (Anthropic API)
- **Agent Framework**: Claude Agent SDK
- **MCP**: Model Context Protocol
- **Embedding**: sentence-transformers (multilingual-e5-base)
- **ベクトルDB**: ChromaDB
- **キーワード検索**: BM25 (rank_bm25)
- **再ランキング**: CrossEncoder (ms-marco-MiniLM-L-6-v2)
- **PDF抽出**: pdfplumber, PyMuPDF, pdfminer.six
- **CLI**: Click, Rich

## MCPツール一覧

MCPサーバーが提供するツール:

| ツール | 機能 |
|--------|------|
| `search_docs` | ドキュメント全文検索（RAG） |
| `get_index_status` | インデックス状態確認 |

## 出力例

### 自律Q&A

```
╭───────────────────────────────────────────────────╮
│ Question: この製品の主な機能は？                    │
╰───────────────────────────────────────────────────╯

[回答]
この製品の主な機能は以下の3つです：
1. データ管理機能
2. レポート生成機能
3. 自動バックアップ機能

[出典]
  製品概要.pdf (V1.0)
  セクション: 1.2, ページ: 5-7
```

## カスタマイズ

### ドキュメントタイプの指定

`--doc-type` オプションで、エージェントのプロンプトに反映されるドキュメントタイプを指定できます:

```bash
python src/main.py agent-qa "エラーコードの意味は？" --doc-type "トラブルシューティングガイド"
```

### チャンクサイズの調整

`src/knowledge/chunker.py` の `Chunker` クラスで、チャンクサイズとオーバーラップを調整できます:

```python
chunker = Chunker(
    fixed_chunk_size=600,    # チャンクの最大文字数
    fixed_chunk_overlap=100,  # チャンク間のオーバーラップ文字数
)
```

## ライセンス

MIT License
