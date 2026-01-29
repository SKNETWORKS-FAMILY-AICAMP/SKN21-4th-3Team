"""
FileName    : bm25.py
Author      : 조남웅
Date        : 2026-01-29
Description : BM25 Retriever (Sparse Ranking Based)
              - Inverted Index 기반
              - distance 개념 제거
              - python -m src.rag.retrievers_test.bm25 실행 지원
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
from typing import Any, Dict, List
from collections import Counter, defaultdict
import math
import re
import time

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
# BM25 Retriever Factory
# -------------------------------------------------------------
def create_bm25_retriever(vector_db, top_k: int = DEFAULT_TOP_K):
    """
    Sparse 기반 BM25 Retriever
    - score 기반 랭킹만 제공
    - DIST 평가 대상 아님
    """

    collection = vector_db.collection
    all_data = collection.get(include=["documents", "metadatas"])

    documents = all_data["documents"]
    metadatas = all_data["metadatas"]
    N = len(documents)

    print(f"[INFO] BM25 indexing {N} documents")

    inverted_index = defaultdict(list)
    doc_lens = []
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

    def retriever(query: str) -> List[Dict[str, Any]]:
        q_tokens = tokenize(query)
        scores = defaultdict(float)

        for t in q_tokens:
            postings = inverted_index.get(t)
            if not postings:
                continue

            idf_t = idf.get(t, 0.0)

            for doc_id, tf in postings:
                numerator = tf * (K1 + 1)
                denominator = tf + K1 * (1 - B + B * (doc_lens[doc_id] / avgdl))
                scores[doc_id] += idf_t * (numerator / denominator)

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {
                "content": documents[doc_id],
                "metadata": metadatas[doc_id],
                "score": score,
            }
            for doc_id, score in ranked
        ]

    print("[INFO] BM25 Retriever ready (Sparse, Ranking only)")
    return retriever


# =============================================================
# Entry Point
# =============================================================
if __name__ == "__main__":
    print("[RUN] BM25 Retriever Test Start")

    test_queries = [
        "요즘 너무 불안해서 잠이 안 와",
        "계속 실패하는 느낌이야",
        "아무것도 하기 싫어",
        "미래가 걱정돼",
        "위로가 필요해"
    ]

    # Load DB
    t0 = time.time()
    vector_db = load_vector_db()
    t1 = time.time()
    print(f"[TIME] DB Loading Time: {t1 - t0:.4f} sec")

    # Create Retriever
    bm25 = create_bm25_retriever(vector_db, top_k=5)

    # Run Tests
    for idx, query in enumerate(test_queries, 1):
        print("\n" + "=" * 60)
        print(f"[QUERY {idx}] {query}")

        qs = time.time()
        results = bm25(query)
        qe = time.time()

        print(f"[TIME] Retrieval Time: {qe - qs:.4f} sec")

        for rank, r in enumerate(results, 1):
            print(f"\n  [{rank}] SCORE = {r['score']:.4f}")
            print(f"      {r['content'][:300]}")
