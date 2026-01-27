"""
FileName    : retriever.py
Auth        : 조남웅
Date        : 2026-01-05
Description : 심리상담 & 명언 챗봇용 Retriever 구현
              VectorDB에서 문서를 검색하여 RAG에 전달하는 역할
Issue/Note  : 초기 구현
              VectorDB(ChromaDB) 정상 로드 및 문서 수 출력 확인
              similarity 기반 Retriever 동작 검증
"""


# -------------------------------------------------------------
# OpenAIConfig Runtime Injection
# -------------------------------------------------------------

import config.db_config as db_config
import os

if not hasattr(db_config, "OpenAIConfig"):
    class OpenAIConfig:
        api_key = os.getenv("OPENAI_API_KEY")
        embedding_model = "text-embedding-3-small"

    db_config.OpenAIConfig = OpenAIConfig


# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

from typing import Any, List, Dict, Optional

from src.database.vector_store import VectorStore


# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------

DEFAULT_TOP_K = 5
DEFAULT_SEARCH_TYPE = "similarity" 


# -------------------------------------------------------------
# VectorDB Load
# -------------------------------------------------------------

def load_vector_db(persist_directory: Optional[str] = None) -> VectorStore:
    """
    VectorStore(ChromaDB) 로드
    - 기존 컬렉션 중 문서가 있는 컬렉션을 사용
    """

    vector_db = VectorStore(persist_directory=persist_directory)

    client = vector_db.client

    collections = client.list_collections()

    if not collections:
        print("[ERROR] No collections found in vector store")
        return vector_db

    # 문서가 있는 컬렉션 찾기
    selected = None
    for col in collections:
        col_obj = client.get_collection(name=col.name)
        count = col_obj.count()
        print(f"[INFO] Found collection: {col.name}, documents: {count}")
        if count > 0 and selected is None:
            selected = col_obj

    if selected:
        vector_db.collection = selected
        print(f"[INFO] Using collection: {selected.name}")
        print(f"[INFO] Total documents: {selected.count()}")
    else:
        print("[WARN] All collections are empty")

    print("[INFO] VectorDB loaded")

    return vector_db


# -------------------------------------------------------------
# Retriever Factory
# -------------------------------------------------------------

def create_retriever(
    vector_db: VectorStore,
    top_k: int = DEFAULT_TOP_K,
):
    """
    Retriever 생성 (함수 기반)

    Args:
        vector_db: VectorStore 인스턴스
        top_k: 검색 결과 개수

    Returns:
        retriever 함수
    """

    def retriever(
        query: str,
        category: Optional[str] = None,
        speaker: Optional[str] = None,
        min_severity: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        실제 검색 수행 함수

        Args:
            query: 사용자 질문
            category: DEPRESSION / ANXIETY / ADDICTION / NORMAL
            speaker: 상담사 / 내담자
            min_severity: 최소 심각도 (0~3)

        Returns:
            [
              {
                "content": "...",
                "metadata": {...},
                "distance": 0.23
              },
              ...
            ]
        """

        # -----------------------------
        # metadata 필터 구성
        # -----------------------------
        where = {}

        if category:
            where["category"] = category

        if speaker:
            where["speaker"] = speaker

        if min_severity is not None:
            where["severity"] = {"$gte": min_severity}

        # -----------------------------
        # Vector 검색
        # -----------------------------
        results = vector_db.search(
            query=query,
            n_results=top_k,
            where=where if where else None
        )

        # -----------------------------
        # 결과 정리
        # -----------------------------
        formatted_results = []
        for i, doc in enumerate(results["documents"]):
            formatted_results.append({
                "content": doc,
                "metadata": results["metadatas"][i] if results["metadatas"] else {},
                "distance": results["distances"][i] if results["distances"] else None,
            })

        return formatted_results

    print("[INFO] Retriever created")
    print(f"       top_k = {top_k}")

    return retriever


# -------------------------------------------------------------
# Debug Functions
# -------------------------------------------------------------

def debug_retriever(retriever, query: str):
    """
    Retriever 동작 확인용 Debug 함수
    """
    print("\n[DEBUG] Retriever Test")
    print(f"[DEBUG] Query: {query}")

    results = retriever(query)

    print(f"[DEBUG] Retrieved documents count: {len(results)}")

    for idx, r in enumerate(results[:3]):
        print(f"\n[DEBUG] Document {idx + 1}")
        print("-" * 40)
        print(r["content"][:300])
        print("[META]", r["metadata"])
        print("[DIST]", r["distance"])


# -------------------------------------------------------------
# Main (Standalone Test)
# -------------------------------------------------------------

def main():
    # 1. VectorDB 로드
    vector_db = load_vector_db()

    # 2. Retriever 생성
    retriever = create_retriever(
        vector_db=vector_db,
        top_k=DEFAULT_TOP_K
    )

    # 3. Debug 테스트
    test_queries = [
        "요즘 너무 불안해서 잠이 안 와",
        "계속 실패하는 느낌이야",
        "위로가 필요해"
    ]

    for query in test_queries:
        debug_retriever(retriever, query)


# -------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------

if __name__ == "__main__":
    main()