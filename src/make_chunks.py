import json
import re

MARKDOWN_FILE = "data/syllabus.md"
CHUNKS_FILE = "data/chunks.json"


def load_markdown(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_by_section(text):
    chunks = []
    sections = re.split(r"\n(?=## )", text.strip())

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


if __name__ == "__main__":
    text = load_markdown(MARKDOWN_FILE)
    chunks = chunk_by_section(text)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"총 {len(chunks)}개의 청크 생성 완료")
    for c in chunks:
        print(f"  {c['chunk_id']}: {c['section']} ({len(c['text'])}자)")
