"""PDF抽出モジュール - マルチエクストラクタ実装

pdfplumber → PyMuPDF → エラーログ の優先順位でフォールバック。
pdfminer.six でフォント情報を並行取得し、見出しレベルを付与。
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTChar, LTTextContainer

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPage:
    """抽出されたページの情報"""
    file: str
    version: str
    page: int
    section: str
    section_level: int
    text: str
    tables: list = field(default_factory=list)


@dataclass
class FontInfo:
    """フォント情報"""
    size: float
    is_bold: bool
    text: str


class PDFExtractor:
    """マルチエクストラクタによるPDF抽出"""

    # 文字化け検出用: 制御文字の割合閾値
    GARBLE_THRESHOLD = 0.1
    # 見出し検出用: フォントサイズ閾値（本文より大きいものを見出しとする）
    HEADING_FONT_SIZE_THRESHOLD = 12.0

    def __init__(self):
        self._current_section = ""
        self._current_section_level = 0

    def extract(self, pdf_path: Path) -> list[ExtractedPage]:
        """PDFからページ単位でテキストを抽出"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        version = self._extract_version(pdf_path)
        font_info_by_page = self._extract_font_info(pdf_path)

        pages = []
        try:
            pages = self._extract_with_pdfplumber(pdf_path, version, font_info_by_page)
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, trying PyMuPDF")
            try:
                pages = self._extract_with_pymupdf(pdf_path, version, font_info_by_page)
            except Exception as e2:
                logger.error(f"PyMuPDF also failed: {e2}")
                raise

        return pages

    def _extract_version(self, pdf_path: Path) -> str:
        """ファイル名またはメタデータからバージョンを抽出"""
        # ファイル名からバージョンを抽出 (例: Document_V15.0_xxx.pdf)
        filename = pdf_path.stem
        version_match = re.search(r'V?(\d+\.\d+)', filename, re.IGNORECASE)
        if version_match:
            return f"V{version_match.group(1)}"

        # メタデータから抽出を試みる
        try:
            with fitz.open(pdf_path) as doc:
                metadata = doc.metadata
                if metadata:
                    for key in ['subject', 'title', 'keywords']:
                        value = metadata.get(key, '')
                        if value:
                            match = re.search(r'V?(\d+\.\d+)', value, re.IGNORECASE)
                            if match:
                                return f"V{match.group(1)}"
        except Exception:
            pass

        return "unknown"

    def _extract_font_info(self, pdf_path: Path) -> dict[int, list[FontInfo]]:
        """pdfminer.sixでフォント情報を抽出"""
        font_info_by_page: dict[int, list[FontInfo]] = {}

        try:
            laparams = LAParams(
                line_margin=0.5,
                word_margin=0.1,
                char_margin=2.0,
                boxes_flow=0.5,
            )

            for page_num, page_layout in enumerate(extract_pages(pdf_path, laparams=laparams), 1):
                font_info_by_page[page_num] = []

                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        for text_line in element:
                            text = ""
                            max_size = 0.0
                            is_bold = False

                            for char in text_line:
                                if isinstance(char, LTChar):
                                    text += char.get_text()
                                    if char.size > max_size:
                                        max_size = char.size
                                    fontname = char.fontname.lower() if char.fontname else ""
                                    if 'bold' in fontname or 'heavy' in fontname:
                                        is_bold = True

                            if text.strip():
                                font_info_by_page[page_num].append(
                                    FontInfo(size=max_size, is_bold=is_bold, text=text.strip())
                                )
        except Exception as e:
            logger.warning(f"Failed to extract font info: {e}")

        return font_info_by_page

    def _detect_section(self, text: str, font_info: Optional[FontInfo]) -> tuple[str, int]:
        """テキストから節番号・見出しを検出"""
        # 番号付き見出しパターン (例: "3.2.1 コマンド名")
        heading_pattern = r'^(\d+(?:\.\d+)*)\s+(.+)$'
        match = re.match(heading_pattern, text.strip())

        if match:
            section_num = match.group(1)
            section_title = match.group(2)
            level = len(section_num.split('.'))
            return f"{section_num} {section_title}", level

        # フォント情報から見出しを判定
        if font_info and (font_info.size >= self.HEADING_FONT_SIZE_THRESHOLD or font_info.is_bold):
            # 大きいフォントまたは太字は見出しの可能性
            if len(text.strip()) < 100:  # 短いテキストは見出しの可能性が高い
                return text.strip(), 1

        return "", 0

    def _is_garbled(self, text: str) -> bool:
        """文字化けを検出"""
        if not text:
            return True

        # 制御文字の割合をチェック
        control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
        ratio = control_chars / len(text) if text else 0

        return ratio > self.GARBLE_THRESHOLD

    def _extract_with_pdfplumber(
        self,
        pdf_path: Path,
        version: str,
        font_info_by_page: dict[int, list[FontInfo]]
    ) -> list[ExtractedPage]:
        """pdfplumberでテキストを抽出"""
        pages = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""

                if self._is_garbled(text):
                    raise ValueError(f"Garbled text detected on page {page_num}")

                # テーブルを抽出
                tables = []
                try:
                    extracted_tables = page.extract_tables()
                    if extracted_tables:
                        tables = extracted_tables
                except Exception:
                    pass

                # 見出しを検出
                font_infos = font_info_by_page.get(page_num, [])
                for fi in font_infos:
                    section, level = self._detect_section(fi.text, fi)
                    if section:
                        self._current_section = section
                        self._current_section_level = level
                        break

                pages.append(ExtractedPage(
                    file=pdf_path.name,
                    version=version,
                    page=page_num,
                    section=self._current_section,
                    section_level=self._current_section_level,
                    text=text,
                    tables=tables,
                ))

        return pages

    def _extract_with_pymupdf(
        self,
        pdf_path: Path,
        version: str,
        font_info_by_page: dict[int, list[FontInfo]]
    ) -> list[ExtractedPage]:
        """PyMuPDFでテキストを抽出（フォールバック）"""
        pages = []

        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()

                if self._is_garbled(text):
                    logger.warning(f"Garbled text on page {page_num}, skipping")
                    continue

                # 見出しを検出
                font_infos = font_info_by_page.get(page_num, [])
                for fi in font_infos:
                    section, level = self._detect_section(fi.text, fi)
                    if section:
                        self._current_section = section
                        self._current_section_level = level
                        break

                pages.append(ExtractedPage(
                    file=pdf_path.name,
                    version=version,
                    page=page_num,
                    section=self._current_section,
                    section_level=self._current_section_level,
                    text=text,
                    tables=[],  # PyMuPDFではテーブル抽出は簡略化
                ))

        return pages
