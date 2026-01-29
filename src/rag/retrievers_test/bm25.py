"""
FileName    : bm25.py
Auth        : 조남웅
Date        : 2026-01-29
Description : BM25 Retriever (Inverted Index Optimized)
              - Full document scan 제거
              - Token 기반 역색인으로 시밀러급 속도 달성
Issue/Note  :
  - 문서 수 제한 없음
  - Pure Python BM25의 구조적 한계 해결
  - Similarity Retriever와 동일한 출력 포맷 유지
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
from typing import Any, Dict, List, Optional
from collections import Counter, defaultdict
import math
import re
import time

from src.database.vector_store import VectorStore
from src.rag.retriever import load_vector_db

# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------
DEFAULT_TOP_K = 5

K1 = 1.5
B = 0.75

_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]+")

# -------------------------------------------------------------
# Utils
# -------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]

# -------------------------------------------------------------
# BM25 Retriever Factory (Inverted Index)
# -------------------------------------------------------------
def create_bm25_retriever(
    vector_db: VectorStore,
    top_k: int = DEFAULT_TOP_K,
):
    """
    Inverted-Index 기반 BM25 Retriever
    """

    collection = vector_db.collection
    all_data = collection.get(include=["documents", "metadatas"])

    documents: List[str] = all_data["documents"]
    metadatas: List[dict] = all_data["metadatas"]

    N = len(documents)
    print(f"[INFO] BM25 indexing {N} documents (no truncation)")

    # ---------------------------------------------------------
    # 1. Index Construction
    # ---------------------------------------------------------
    inverted_index = defaultdict(list)   # token -> [(doc_id, tf)]
    doc_lens: List[int] = []
    df = defaultdict(int)

    for doc_id, doc in enumerate(documents):
        tokens = tokenize(doc)
        tf = Counter(tokens)

        doc_lens.append(len(tokens))

        for t, freq in tf.items():
            inverted_index[t].append((doc_id, freq))

        for t in tf.keys():
            df[t] += 1

    avgdl = sum(doc_lens) / N

    idf = {
        t: math.log(1 + (N - df_t + 0.5) / (df_t + 0.5))
        for t, df_t in df.items()
    }

    print(f"[INFO] BM25 index built | vocab size = {len(inverted_index)}")

    # ---------------------------------------------------------
    # 2. Retriever
    # ---------------------------------------------------------
    def retriever(
        query: str,
        category: Optional[str] = None,
        speaker: Optional[str] = None,
        min_severity: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        q_tokens = tokenize(query)
        scores = defaultdict(float)

        for t in q_tokens:
            postings = inverted_index.get(t)
            if not postings:
                continue

            idf_t = idf.get(t, 0.0)

            for doc_id, tf in postings:
                meta = metadatas[doc_id]

                if category and meta.get("category") != category:
                    continue
                if speaker and meta.get("speaker") != speaker:
                    continue
                if min_severity is not None and meta.get("severity", 0) < min_severity:
                    continue

                numerator = tf * (K1 + 1)
                denominator = tf + K1 * (1 - B + B * (doc_lens[doc_id] / avgdl))
                scores[doc_id] += idf_t * (numerator / denominator)

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        max_score = ranked[0][1]

        results = []
        for doc_id, score in ranked:
            if score <= 0:
                continue

            results.append({
                "content": documents[doc_id],
                "metadata": metadatas[doc_id],
                "distance": 1.0 - (score / max_score),
            })

        return results

    print("[INFO] BM25 Retriever ready (Inverted Index)")
    return retriever

# -------------------------------------------------------------
# Debug / Standalone Test
# -------------------------------------------------------------
def main():
    start_total = time.time()
    print("[INFO] BM25 Retriever Test Start")

    vector_db = load_vector_db()
    load_end = time.time()
    print(f"[TIME] VectorDB 로딩: {load_end - start_total:.4f}초")
    
    retriever = create_bm25_retriever(vector_db)

    test_queries = [
        "요즘 너무 불안해서 잠이 안 와",
        "계속 실패하는 느낌이야",
        "아무것도 하기 싫어",
        "미래가 걱정돼",
        "위로가 필요해",
    ]

    for q in test_queries:
        print("\n" + "=" * 80)
        print(f"[QUERY] {q}")

        start_time = time.time()
        results = retriever(q)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"[TIME]: {elapsed_time:.4f}초")

        for i, r in enumerate(results):
            print("\n----------------------------------------")
            print(r["content"][:300])
            print("[META]", r["metadata"])
            print("[DIST]", round(r["distance"], 6))
            print(f"[DEBUG] Document {i+1}")
    
    end_total = time.time()
    print(f"\n[TIME] 전체 실행 시간: {end_total - start_total:.4f}초")

# -------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
