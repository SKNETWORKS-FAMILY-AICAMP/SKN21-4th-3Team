"""
FileName    : hybrid.py
Auth        : 조남웅
Date        : 2026-01-29
Description : Similarity + Contextual Hybrid Retriever (정석 버전)
Issue/Note  :
  - Hybrid = Similarity(Dense) + Contextual
  - 상담 도메인 정합성 + 속도 최적화
  - TIME 측정 포함
  - python -m src.rag.retrievers_test.hybrid 실행 지원
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
from typing import Any, Dict, List, Optional, Tuple, Set
import time

from src.rag.retriever import load_vector_db, create_retriever
from src.database.vector_store import VectorStore


# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------
DEFAULT_TOP_K = 5
DEFAULT_SEED_K = 3
DEFAULT_WINDOW = 1
USE_BEST_SESSION_ONLY = True


# -------------------------------------------------------------
# Contextual Helpers
# -------------------------------------------------------------
def _extract_session_turn(meta: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    if not meta:
        return None, None
    sid = meta.get("session_id")
    tid = meta.get("turn_index")
    try:
        tid = int(tid) if tid is not None else None
    except Exception:
        tid = None
    return sid, tid


def _get_doc_by_session_turn(
    vector_db: VectorStore,
    session_id: str,
    turn_index: int,
):
    try:
        got = vector_db.collection.get(
            where={"session_id": session_id, "turn_index": turn_index},
            include=["documents", "metadatas"],
        )
    except Exception:
        return None

    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    if not docs:
        return None

    return docs[0], metas[0]


def _distance_to_query(vector_db: VectorStore, query: str, sid: str, tid: int):
    try:
        r = vector_db.collection.query(
            query_texts=[query],
            n_results=1,
            where={"session_id": sid, "turn_index": tid},
        )
        d = (r.get("distances") or [[]])[0]
        if d:
            return float(d[0])
    except Exception:
        pass
    return None


# -------------------------------------------------------------
# Hybrid Retriever Factory
# -------------------------------------------------------------
def create_similarity_contextual_hybrid(
    vector_db: VectorStore,
    top_k: int = DEFAULT_TOP_K,
    seed_k: int = DEFAULT_SEED_K,
    window: int = DEFAULT_WINDOW,
):
    """
    Similarity + Contextual Hybrid Retriever
    """

    # 1️⃣ Similarity Retriever (Dense)
    similarity_retriever = create_retriever(
        vector_db=vector_db,
        top_k=seed_k,
    )

    def retriever(query: str) -> List[Dict[str, Any]]:
        # -----------------------------
        # 1) Similarity Seed
        # -----------------------------
        seeds = similarity_retriever(query=query) or []
        if not seeds:
            return []

        enriched = []
        for s in seeds:
            sid, tid = _extract_session_turn(s.get("metadata", {}))
            if sid is None or tid is None:
                continue
            enriched.append({
                "content": s["content"],
                "metadata": s["metadata"],
                "distance": s["distance"],
                "sid": sid,
                "tid": tid,
            })

        if not enriched:
            return seeds[:top_k]

        enriched.sort(key=lambda x: x["distance"])
        best_sid = enriched[0]["sid"]

        target_sids = {best_sid} if USE_BEST_SESSION_ONLY else {e["sid"] for e in enriched}

        # -----------------------------
        # 2) Contextual Expansion
        # -----------------------------
        collected = {}
        for e in enriched:
            if e["sid"] not in target_sids:
                continue

            for tt in range(e["tid"] - window, e["tid"] + window + 1):
                if tt < 0:
                    continue
                key = (e["sid"], tt)
                if key in collected:
                    continue

                got = _get_doc_by_session_turn(vector_db, e["sid"], tt)
                if not got:
                    continue

                doc, meta = got
                dist = _distance_to_query(vector_db, query, e["sid"], tt)

                collected[key] = {
                    "content": doc,
                    "metadata": meta,
                    "distance": dist,
                    "_turn": tt,
                }

        if not collected:
            return seeds[:top_k]

        # -----------------------------
        # 3) 정렬 및 반환
        # -----------------------------
        items = list(collected.values())
        items.sort(key=lambda x: x["_turn"])

        if len(items) > top_k:
            center = enriched[0]["tid"]
            items.sort(key=lambda x: abs(x["_turn"] - center))
            items = items[:top_k]
            items.sort(key=lambda x: x["_turn"])

        return [
            {
                "content": it["content"],
                "metadata": it["metadata"],
                "distance": it["distance"],
            }
            for it in items
        ]

    print("[INFO] Hybrid Retriever created: Similarity + Contextual")
    return retriever


# -------------------------------------------------------------
# Main (Standalone Test + TIME)
# -------------------------------------------------------------
def main():
    total_start = time.time()

    # VectorDB 로딩
    t0 = time.time()
    vector_db = load_vector_db()
    t1 = time.time()
    print(f"[TIME] VectorDB 로딩: {t1 - t0:.4f}초")

    retriever = create_similarity_contextual_hybrid(
        vector_db=vector_db,
        top_k=DEFAULT_TOP_K,
        seed_k=DEFAULT_SEED_K,
        window=DEFAULT_WINDOW,
    )

    queries = [
        "요즘 너무 불안해서 잠이 안 와",
        "계속 실패하는 느낌이야",
        "미래가 걱정돼",
        "위로가 필요해",
    ]

    for q in queries:
        print("\n" + "=" * 80)
        print(f"[QUERY] {q}")
        qs = time.time()
        results = retriever(q)
        qe = time.time()
        print(f"[TIME]: {qe - qs:.4f}초")

        for i, r in enumerate(results):
            print(f"\n[DEBUG] Doc {i+1}")
            print((r["content"] or "")[:300])
            print("[DIST]", r["distance"])

    total_end = time.time()
    print(f"\n[TIME] 전체 실행 시간: {total_end - total_start:.4f}초")


# -------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
