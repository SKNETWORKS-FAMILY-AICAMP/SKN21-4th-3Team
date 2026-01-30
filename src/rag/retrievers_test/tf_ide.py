"""
FileName    : tf_ide.py
Author      : 조남웅
Date        : 2026-01-29
Description : TF-IDF Retriever (Sparse Ranking Based)
              - Inverted Index 기반
              - distance 개념 제거
              - BM25와 동일한 랭킹 평가 위치
              - python -m src.rag.retrievers_test.tf_ide 실행 지원
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
from typing import Any, Dict, List, Optional
from collections import Counter, defaultdict
import math
import re
import time

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
# TF-IDF Retriever Factory (Sparse, Ranking Based)
# -------------------------------------------------------------
def create_tfidf_retriever(
    vector_db,
    top_k: int = DEFAULT_TOP_K,
):
    """
    Sparse 기반 TF-IDF Retriever
    - cosine similarity score 기반 랭킹
    - DIST 개념 없음
    """

    collection = vector_db.collection
    all_data = collection.get(include=["documents", "metadatas"])

    documents: List[str] = all_data["documents"]
    metadatas: List[dict] = all_data["metadatas"]

    N = len(documents)
    print(f"[INFO] TF-IDF indexing {N} documents")

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
    # 3. Retriever (Ranking Only)
    # ---------------------------------------------------------
    def retriever(query: str) -> List[Dict[str, Any]]:
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
                scores[doc_id] += q_weight * (tf * idf[t])

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, dot in ranked:
            if doc_norm[doc_id] == 0:
                continue

            sim = dot / (q_norm * doc_norm[doc_id])
            results.append({
                "content": documents[doc_id],
                "metadata": metadatas[doc_id],
                "score": sim,   # ✅ 랭킹 점수 (클수록 좋음)
            })

        return results

    print("[INFO] TF-IDF Retriever ready (Sparse, Ranking only)")
    return retriever


# =============================================================
# Entry Point (python -m 실행용)
# =============================================================
if __name__ == "__main__":
    print("[RUN] TF-IDF Retriever Test Start")

    test_queries = [
        "요즘 너무 불안해서 잠이 안 와",
        "계속 실패하는 느낌이야",
        "아무것도 하기 싫어",
        "미래가 걱정돼",
        "위로가 필요해",
    ]

    # Load Vector DB
    t0 = time.time()
    vector_db = load_vector_db()
    t1 = time.time()
    print(f"[TIME] DB Loading Time: {t1 - t0:.4f} sec")

    # Create Retriever
    tfidf = create_tfidf_retriever(vector_db, top_k=5)

    # Run Tests
    for idx, query in enumerate(test_queries, 1):
        print("\n" + "=" * 60)
        print(f"[QUERY {idx}] {query}")

        qs = time.time()
        results = tfidf(query)
        qe = time.time()

        print(f"[TIME] Retrieval Time: {qe - qs:.4f} sec")

        for rank, r in enumerate(results, 1):
            print(f"\n  [{rank}] SCORE = {r['score']:.6f}")
            print(f"      {r['content'][:300]}")
