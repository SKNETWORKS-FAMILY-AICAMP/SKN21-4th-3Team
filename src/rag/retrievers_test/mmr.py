"""
FileName    : mmr.py
Auth        : 조남웅
Date        : 2026-01-06
Description : 심리상담 & 명언 챗봇용 MMR Retriever 구현
              similarity 검색 결과를 기반으로
              MMR(Max Marginal Relevance) 방식으로 문서를 재선정
Issue/Note  : VectorStore는 그대로 두고 Retriever 단계에서 MMR 적용
"""

# -------------------------------------------------------------
# OpenAIConfig Runtime Injection
# -------------------------------------------------------------

import os
import config.db_config as db_config

if not hasattr(db_config, "OpenAIConfig"):
    class OpenAIConfig:
        api_key = os.getenv("OPENAI_API_KEY")
        embedding_model = "text-embedding-3-small"

    db_config.OpenAIConfig = OpenAIConfig


# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

from typing import Any, List, Dict, Optional
import time

from src.database.vector_store import VectorStore
from src.rag.retriever import load_vector_db  # 컬렉션 자동 선택 로딩 함수 재사용


# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------

DEFAULT_TOP_K = 5
DEFAULT_FETCH_K = 40
DEFAULT_LAMBDA = 0.4


# -------------------------------------------------------------
# MMR Utility
# -------------------------------------------------------------

def _mmr_select(
    distances: List[float],
    top_k: int,
    lambda_mult: float
) -> List[int]:
    """
    MMR 인덱스 선택 로직

    - distances: query와 문서 간 거리 (작을수록 유사)
    - relevance: -distance (distance가 작을수록 relevance 높음)
    - diversity: 이미 선택된 문서들과의 거리 차이로 근사
    """
    if not distances:
        return []

    selected: List[int] = []
    candidates = list(range(len(distances)))

    # 1) 가장 유사한 문서 먼저 선택
    first = min(candidates, key=lambda i: distances[i])
    selected.append(first)
    candidates.remove(first)

    # 2) 이후 문서 선택
    while len(selected) < min(top_k, len(distances)) and candidates:
        best_idx = None
        best_score = float("-inf")

        for i in candidates:
            relevance = -distances[i]
            diversity = min(abs(distances[i] - distances[j]) for j in selected)

            score = lambda_mult * relevance + (1 - lambda_mult) * diversity

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            break

        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected


# -------------------------------------------------------------
# Retriever Factory
# -------------------------------------------------------------

def create_mmr_retriever(
    vector_db: VectorStore,
    top_k: int = DEFAULT_TOP_K,
    fetch_k: int = DEFAULT_FETCH_K,
    lambda_mult: float = DEFAULT_LAMBDA,
):
    """
    MMR 기반 Retriever 생성
    """

    def retriever(
        query: str,
        category: Optional[str] = None,
        speaker: Optional[str] = None,
        min_severity: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        # -----------------------------
        # metadata 필터 구성
        # -----------------------------
        where: Dict[str, Any] = {}

        # 주의: 사장님 데이터 메타 키는 default_category, category_paragraph_speaker 등일 수 있습니다.
        # 기존 코드 호환을 위해 일단 category/speaker 키를 유지합니다.
        if category:
            where["category"] = category
        if speaker:
            where["speaker"] = speaker
        if min_severity is not None:
            where["severity"] = {"$gte": min_severity}

        # -----------------------------
        # 1) similarity 기반 후보 검색
        # -----------------------------
        base_results = vector_db.search(
            query=query,
            n_results=fetch_k,
            where=where if where else None
        )

        documents = base_results.get("documents", []) or []
        metadatas = base_results.get("metadatas", []) or []
        distances = base_results.get("distances", []) or []

        if not documents:
            return []

        # -----------------------------
        # 2) MMR 재선정
        # -----------------------------
        selected_indices = _mmr_select(
            distances=distances,
            top_k=top_k,
            lambda_mult=lambda_mult
        )

        # -----------------------------
        # 3) 결과 정리 (similarity와 동일 포맷)
        # -----------------------------
        formatted_results: List[Dict[str, Any]] = []
        for i in selected_indices:
            formatted_results.append({
                "content": documents[i],
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": distances[i],
            })

        return formatted_results

    print("[INFO] MMR Retriever created")
    print(f"       top_k       = {top_k}")
    print(f"       fetch_k     = {fetch_k}")
    print(f"       lambda_mult = {lambda_mult}")

    return retriever


# -------------------------------------------------------------
# Debug Functions
# -------------------------------------------------------------

def debug_retriever(retriever, query: str):
    """
    Retriever 동작 확인용 Debug 함수
    """
    print("\n[DEBUG] Retriever Test (MMR)")
    print(f"[DEBUG] Query: {query}")

    start_time = time.time()
    results = retriever(query)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"[DEBUG] Retrieved documents count: {len(results)}")
    print(f"[TIME]: {elapsed_time:.4f}초")

    for idx, r in enumerate(results[:3]):
        print(f"\n[DEBUG] Document {idx + 1}")
        print("-" * 40)
        print((r.get("content") or "")[:300])
        print("[META]", r.get("metadata"))
        print("[DIST]", r.get("distance"))


# -------------------------------------------------------------
# Main (Standalone Test)
# -------------------------------------------------------------

def main():
    start_total = time.time()
    
    # retriever.py의 load_vector_db를 사용하여
    # 문서가 존재하는 컬렉션을 자동으로 선택합니다.
    vector_db = load_vector_db()
    load_end = time.time()
    print(f"[TIME] VectorDB 로딩: {load_end - start_total:.4f}초")

    retriever = create_mmr_retriever(
        vector_db=vector_db,
        top_k=DEFAULT_TOP_K,
        fetch_k=DEFAULT_FETCH_K,
        lambda_mult=DEFAULT_LAMBDA
    )

    test_queries = [
        "요즘 너무 불안해서 잠이 안 와",
        "계속 실패하는 느낌이야",
        "아무것도 하기 싫어",
        "미래가 걱정돼",
        "위로가 필요해"
    ]

    for query in test_queries:
        debug_retriever(retriever, query)
    
    end_total = time.time()
    print(f"\n[TIME] 전체 실행 시간: {end_total - start_total:.4f}초")


# -------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------

if __name__ == "__main__":
    main()
