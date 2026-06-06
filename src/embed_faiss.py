import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "data/chunks.json"
INDEX_FILE = "data/bge.index"
EMBEDDING_MODEL = "BAAI/bge-m3"


def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(chunks, model):
    texts = [c["text"] for c in chunks]
    print(f"임베딩 생성 중... (모델: {EMBEDDING_MODEL})")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings


if __name__ == "__main__":
    print("청크 로드 중...")
    chunks = load_chunks(CHUNKS_FILE)
    print(f"총 {len(chunks)}개 청크 로드됨")

    print("모델 로드 중...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    index, embeddings = build_index(chunks, model)

    faiss.write_index(index, INDEX_FILE)
    print(f"FAISS 인덱스 저장 완료: {INDEX_FILE}")
    print(f"벡터 차원: {embeddings.shape[1]}, 벡터 수: {index.ntotal}")
