"""チャンク化モジュール

固定長チャンクと見出し単位チャンクを並行して生成する。
"""

import uuid
from dataclasses import dataclass
from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .extractor import ExtractedPage


@dataclass
class Chunk:
    """チャンク情報"""
    chunk_id: str
    chunk_type: Literal["fixed", "heading"]
    file: str
    version: str
    page_start: int
    page_end: int
    section: str
    section_path: str
    text: str


class Chunker:
    """テキストチャンク化"""

    def __init__(
        self,
        fixed_chunk_size: int = 600,
        fixed_chunk_overlap: int = 100,
    ):
        self.fixed_splitter = RecursiveCharacterTextSplitter(
            chunk_size=fixed_chunk_size,
            chunk_overlap=fixed_chunk_overlap,
            separators=["\n\n", "\n", "。", "、", " ", ""],
        )

    def chunk_pages(self, pages: list[ExtractedPage]) -> list[Chunk]:
        """ページリストからチャンクを生成"""
        chunks = []

        # 固定長チャンク生成
        chunks.extend(self._create_fixed_chunks(pages))

        # 見出し単位チャンク生成
        chunks.extend(self._create_heading_chunks(pages))

        return chunks

    def _create_fixed_chunks(self, pages: list[ExtractedPage]) -> list[Chunk]:
        """固定長チャンクを生成"""
        chunks = []

        for page in pages:
            if not page.text.strip():
                continue

            split_texts = self.fixed_splitter.split_text(page.text)

            for text in split_texts:
                if not text.strip():
                    continue

                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    chunk_type="fixed",
                    file=page.file,
                    version=page.version,
                    page_start=page.page,
                    page_end=page.page,
                    section=page.section,
                    section_path=self._build_section_path(page.section),
                    text=text,
                ))

        return chunks

    def _create_heading_chunks(self, pages: list[ExtractedPage]) -> list[Chunk]:
        """見出し単位チャンクを生成

        同じセクションに属するページをまとめて1つのチャンクにする。
        """
        chunks = []
        current_section = ""
        current_texts: list[str] = []
        current_pages: list[ExtractedPage] = []

        for page in pages:
            # セクションが変わったら、前のセクションをチャンクとして保存
            if page.section and page.section != current_section:
                if current_texts and current_section:
                    chunks.append(self._create_heading_chunk(
                        current_section,
                        current_texts,
                        current_pages,
                    ))

                current_section = page.section
                current_texts = []
                current_pages = []

            current_texts.append(page.text)
            current_pages.append(page)

        # 最後のセクションを保存
        if current_texts and current_section:
            chunks.append(self._create_heading_chunk(
                current_section,
                current_texts,
                current_pages,
            ))

        return chunks

    def _create_heading_chunk(
        self,
        section: str,
        texts: list[str],
        pages: list[ExtractedPage],
    ) -> Chunk:
        """見出し単位チャンクを作成"""
        combined_text = "\n\n".join(texts)

        # チャンクが大きすぎる場合は切り詰める（最大3000文字）
        max_length = 3000
        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length] + "..."

        return Chunk(
            chunk_id=str(uuid.uuid4()),
            chunk_type="heading",
            file=pages[0].file if pages else "",
            version=pages[0].version if pages else "",
            page_start=pages[0].page if pages else 0,
            page_end=pages[-1].page if pages else 0,
            section=section,
            section_path=self._build_section_path(section),
            text=combined_text,
        )

    def _build_section_path(self, section: str) -> str:
        """セクションパスを構築（例: "3.2.1 コマンド" → "3 > 3.2 > 3.2.1"）"""
        if not section:
            return ""

        # セクション番号を抽出
        parts = section.split()
        if not parts:
            return ""

        section_num = parts[0]
        num_parts = section_num.split('.')

        # パスを構築
        path_parts = []
        for i in range(1, len(num_parts) + 1):
            path_parts.append('.'.join(num_parts[:i]))

        return " > ".join(path_parts)
