import os
import xml.etree.ElementTree as ET
import zipfile

from docx import Document
from pptx import Presentation
from pypdf import PdfReader
from striprtf.striprtf import rtf_to_text


def _table_to_text(table) -> str:
    """Read a docx table as one line per row, dropping repeated text from merged cells."""
    lines = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            text = cell.text.strip()
            # A horizontally merged cell repeats its text for every column it spans
            if text and (not cells or text != cells[-1]):
                cells.append(text)
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _slide_to_text(slide) -> str:
    """Read every text-bearing shape on a slide, one line per shape."""
    lines = []
    for shape in slide.shapes:
        if shape.has_text_frame and shape.text_frame.text.strip():
            lines.append(shape.text_frame.text.strip())
        elif shape.has_table:
            for row in shape.table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    lines.append(" | ".join(cells))
    return "\n".join(lines)


def _read_text_file(file_path: str) -> str:
    """Read a plain text file, tolerating common non-UTF-8 encodings (e.g. Korean cp949)."""
    with open(file_path, "rb") as f:
        raw = f.read()
    for encoding in ("utf-8", "cp949"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _extract_xml_paragraphs(archive: zipfile.ZipFile, xml_names: list[str], paragraph_tag_suffixes) -> str:
    """Extract text from XML entries inside a zip-based document, one line per paragraph element."""
    lines = []
    for name in xml_names:
        try:
            root = ET.fromstring(archive.read(name))
        except (KeyError, ET.ParseError):
            continue
        for elem in root.iter():
            if elem.tag.endswith(paragraph_tag_suffixes):
                text = "".join(elem.itertext()).strip()
                if text:
                    lines.append(text)
    return "\n".join(lines)


def _extract_odt(file_path: str) -> str:
    """Extract text from an OpenDocument Text (.odt) file's content.xml."""
    with zipfile.ZipFile(file_path) as archive:
        return _extract_xml_paragraphs(archive, ["content.xml"], ("}p", "}h"))


def _extract_hwpx(file_path: str) -> str:
    """Extract text from a Hangul HWPX file's section XML files (Contents/section*.xml)."""
    with zipfile.ZipFile(file_path) as archive:
        section_names = sorted(
            name for name in archive.namelist() if name.startswith("Contents/section") and name.endswith(".xml")
        )
        return _extract_xml_paragraphs(archive, section_names, ("}p",))


def extract_text(file_path: str) -> str:
    """Extract text from a .docx, .pptx, .pdf, .odt, .hwpx, .rtf, or .txt file, chosen by its extension."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".docx":
        document = Document(file_path)
        paragraph_text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        table_text = "\n".join(_table_to_text(table) for table in document.tables)
        return "\n".join(part for part in (paragraph_text, table_text) if part)

    if ext == ".pptx":
        presentation = Presentation(file_path)
        return "\n".join(_slide_to_text(slide) for slide in presentation.slides)

    if ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == ".odt":
        return _extract_odt(file_path)

    if ext == ".hwpx":
        return _extract_hwpx(file_path)

    if ext == ".rtf":
        with open(file_path, "rb") as f:
            raw = f.read()
        return rtf_to_text(raw.decode("latin-1"))

    if ext == ".txt":
        return _read_text_file(file_path)

    raise ValueError(f"Unsupported file type: {ext}")
