import json
import re
import os
from pathlib import Path

DATA_DIR = "data"
CHUNKS_FILE = "data/chunks.json"


def load_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def load_markdown(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_documents():
    """data/ 폴더에서 PDF 및 Markdown 파일을 모두 읽어 합친다."""
    texts = []
    data_path = Path(DATA_DIR)

    pdf_files = sorted(data_path.glob("*.pdf"))
    md_files = sorted(data_path.glob("*.md"))

    for pdf in pdf_files:
        print(f"PDF 로드: {pdf.name}")
        texts.append(load_pdf(str(pdf)))

    for md in md_files:
        print(f"Markdown 로드: {md.name}")
        texts.append(load_markdown(str(md)))

    if not texts:
        raise FileNotFoundError(f"data/ 폴더에 PDF 또는 Markdown 파일이 없습니다.")

    return "\n\n".join(texts)


def chunk_by_section(text):
    chunks = []
    sections = re.split(r"\n(?=#{1,3} )", text.strip())

    for section in sections:
        section = section.strip()
        if not section:
            continue

        title_match = re.match(r"#{1,3}\s+(.+)", section)
        title = title_match.group(1).strip() if title_match else "일반"

        chunks.append({
            "chunk_id": f"chunk_{len(chunks):02d}",
            "section": title,
            "text": section,
        })

    return chunks


def chunk_by_paragraph(text, min_len=100):
    """섹션 구분이 없는 PDF용 단락 기반 청킹."""
    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks = []
    for para in paragraphs:
        para = para.strip()
        if len(para) < min_len:
            continue
        chunks.append({
            "chunk_id": f"chunk_{len(chunks):02d}",
            "section": para[:40].replace("\n", " "),
            "text": para,
        })
    return chunks


if __name__ == "__main__":
    text = load_documents()

    # 마크다운 헤더가 있으면 섹션 기반, 없으면 단락 기반 청킹
    if re.search(r"\n#{1,3} ", text):
        chunks = chunk_by_section(text)
    else:
        chunks = chunk_by_paragraph(text)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n총 {len(chunks)}개의 청크 생성 완료")
    for c in chunks:
        print(f"  {c['chunk_id']}: {c['section']} ({len(c['text'])}자)")
