"""
FileName    : tf_idf.py
Auth        : 조남웅
Date        : 2026-01-29
Description : TF-IDF Retriever (Inverted Index Optimized)
              - Full document scan 제거
              - Token 기반 역색인 구조
              - Similarity Retriever와 동일 출력 포맷
Issue/Note  :
  - 문서 수 제한 없음
  - Pure Python TF-IDF의 구조적 성능 문제 해결
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
_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]+")

# -------------------------------------------------------------
# Utils
# -------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]

# -------------------------------------------------------------
# TF-IDF Retriever Factory (Inverted Index)
# -------------------------------------------------------------
def create_tfidf_retriever(
    vector_db: VectorStore,
    top_k: int = DEFAULT_TOP_K,
):
    """
    Inverted-Index 기반 TF-IDF Retriever
    """

    collection = vector_db.collection
    all_data = collection.get(include=["documents", "metadatas"])

    documents: List[str] = all_data["documents"]
    metadatas: List[dict] = all_data["metadatas"]

    N = len(documents)
    print(f"[INFO] TF-IDF indexing {N} documents (no truncation)")

    # ---------------------------------------------------------
    # 1. Index Construction
    # ---------------------------------------------------------
    inverted_index = defaultdict(list)  # token -> [(doc_id, tf)]
    df = defaultdict(int)
    doc_norm = [0.0] * N

    for doc_id, doc in enumerate(documents):
        tokens = tokenize(doc)
        tf = Counter(tokens)

        for t, freq in tf.items():
            inverted_index[t].append((doc_id, freq))

        for t in tf.keys():
            df[t] += 1

    idf = {
        t: math.log((N + 1) / (df_t + 1)) + 1
        for t, df_t in df.items()
    }

    # ---------------------------------------------------------
    # 2. Precompute Document Norms
    # ---------------------------------------------------------
    for t, postings in inverted_index.items():
        idf_t = idf[t]
        for doc_id, tf in postings:
            w = tf * idf_t
            doc_norm[doc_id] += w * w

    doc_norm = [math.sqrt(n) for n in doc_norm]

    print(f"[INFO] TF-IDF index built | vocab size = {len(inverted_index)}")

    # ---------------------------------------------------------
    # 3. Retriever
    # ---------------------------------------------------------
    def retriever(
        query: str,
        category: Optional[str] = None,
        speaker: Optional[str] = None,
        min_severity: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        q_tokens = tokenize(query)
        tf_q = Counter(q_tokens)

        q_vec = {}
        for t, tf in tf_q.items():
            if t in idf:
                q_vec[t] = tf * idf[t]

        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm == 0:
            return []

        scores = defaultdict(float)

        for t, q_weight in q_vec.items():
            postings = inverted_index.get(t)
            if not postings:
                continue

            for doc_id, tf in postings:
                meta = metadatas[doc_id]

                if category and meta.get("category") != category:
                    continue
                if speaker and meta.get("speaker") != speaker:
                    continue
                if min_severity is not None and meta.get("severity", 0) < min_severity:
                    continue

                scores[doc_id] += q_weight * (tf * idf[t])

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in ranked:
            if doc_norm[doc_id] == 0:
                continue

            sim = score / (q_norm * doc_norm[doc_id])
            results.append({
                "content": documents[doc_id],
                "metadata": metadatas[doc_id],
                "distance": 1.0 - sim,
            })

        return results

    print("[INFO] TF-IDF Retriever ready (Inverted Index)")
    return retriever

# -------------------------------------------------------------
# Debug / Standalone Test
# -------------------------------------------------------------
def main():
    start_total = time.time()
    print("[INFO] TF-IDF Retriever Test Start")

    vector_db = load_vector_db()
    load_end = time.time()
    print(f"[TIME] VectorDB 로딩: {load_end - start_total:.4f}초")
    
    retriever = create_tfidf_retriever(vector_db)

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
