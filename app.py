import json
import os
import numpy as np
import faiss
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer

# 기본 설정
CHUNKS_FILE = "data/chunks.json"
INDEX_FILE = "data/bge.index"
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "qwen2.5:3b"
DOC_NAME = "소프트웨어공학_강의계획서.pdf"


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_index_and_chunks():
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks


def search(query, index, chunks, model, top_k=3):
    query_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(chunks):
            results.append({"chunk": chunks[idx], "score": float(score)})
    return results


def generate_answer(query, results):
    context = "\n\n".join(
        [f"[{r['chunk']['section']}]\n{r['chunk']['text']}" for r in results]
    )
    prompt = f"""아래 강의계획서 내용을 참고하여 질문에 한국어로 답하세요. 참고 내용에 없는 것은 모른다고 하세요.

[참고 내용]
{context}

[질문]
{query}

[답변]"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "당신은 소프트웨어공학 강의계획서 기반 질의응답 도우미입니다. 주어진 내용만을 바탕으로 간결하고 정확하게 답하세요.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]


# ── Streamlit UI ──────────────────────────────────────────────
st.set_page_config(
    page_title="소프트웨어공학 강의계획서 RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 소프트웨어공학 강의계획서 RAG 챗봇")
st.caption("INC4119-01 소프트웨어공학 강의계획서 기반 질의응답 | 이서준 / 2021112033")

# 리소스 로드
if not os.path.exists(INDEX_FILE):
    st.error(
        "FAISS 인덱스가 없습니다. 먼저 아래 명령을 실행하세요:\n\n"
        "```\npython src/make_chunks.py\npython src/embed_faiss.py\n```"
    )
    st.stop()

with st.spinner("모델 및 인덱스 로드 중..."):
    embedding_model = load_embedding_model()
    index, chunks = load_index_and_chunks()

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    top_k = st.slider("Top-K 검색 결과", min_value=1, max_value=5, value=3)
    st.divider()
    st.markdown(f"📄 **문서**: {DOC_NAME}")
    st.markdown(f"🗂️ **청크 수**: {len(chunks)}")
    st.markdown(f"🤖 **LLM**: {LLM_MODEL}")
    st.markdown(f"📐 **임베딩**: bge-m3")
    st.divider()
    st.markdown("**예시 질문**")
    examples = [
        "기말고사는 몇 퍼센트인가요?",
        "담당교수 연락처를 알려주세요",
        "8주차 수업 내용이 뭔가요?",
        "강의 목표가 무엇인가요?",
        "14주차에 무엇을 배우나요?",
        "교재가 무엇인가요?",
    ]
    for q in examples:
        if st.button(q, use_container_width=True):
            st.session_state["pending_query"] = q

# 메인 영역
col_main, col_info = st.columns([2, 1])

with col_main:
    st.markdown("### 💬 질문하기")
    default_val = st.session_state.pop("pending_query", "")
    query = st.text_input(
        "질문을 입력하세요...",
        value=default_val,
        placeholder="예: 중간고사는 총 점수의 몇 %인가요?",
        key="query_input",
    )

    ask_btn = st.button("질문하기", type="primary", use_container_width=True)

    if ask_btn and query:
        with st.spinner("🔍 관련 내용 검색 중..."):
            results = search(query, index, chunks, embedding_model, top_k)

        with st.spinner("🤖 답변 생성 중..."):
            answer = generate_answer(query, results)

        st.markdown("### 🤖 답변")
        st.info(answer)

        st.markdown("### 🔍 검색된 근거 (Top-k)")
        for i, r in enumerate(results):
            score = r["score"]
            chunk = r["chunk"]
            with st.expander(
                f"📄 {chunk['chunk_id']} · {chunk['section']} — Score: {score:.4f}",
                expanded=(i == 0),
            ):
                preview = chunk["text"]
                if len(preview) > 600:
                    preview = preview[:600] + "..."
                st.text(preview)

with col_info:
    st.markdown("### ℹ️ 강의 요약")
    st.markdown("""
| 항목 | 내용 |
|------|------|
| 과목 | 소프트웨어공학 |
| 학수번호 | INC4119-01 |
| 학점 | 3학점 |
| 교수 | 김웅섭 |
| 중간고사 | 30% |
| 기말고사 | 30% |
| 과제 | 15% |
| 프로젝트 | 20% |
| 출석 | 5% |
""")
