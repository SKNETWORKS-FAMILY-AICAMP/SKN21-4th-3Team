"""
FileName    : contextual.py
Auth        : 조남웅
Date        : 2026-01-28
Description : 심리상담 & 명언 챗봇용 Contextual Retriever 구현
              similarity 검색 결과(seed)를 기반으로,
              같은 session_id 내에서 turn_index 주변(±window)의 문서를 확장하여 반환
Issue/Note  :
  - VectorStore(ChromaDB) 구조는 그대로 유지하고 Retriever 단계에서 문맥 확장
  - 반환 포맷은 기존 retriever들과 동일하게 유지:
      [{"content": str, "metadata": dict, "distance": float}, ...]
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
from typing import Any, List, Dict, Optional, Tuple, Set
import time

from src.database.vector_store import VectorStore
from src.rag.retriever import load_vector_db  # 빈 컬렉션 방지용


# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------
DEFAULT_TOP_K = 5

# seed(유사도 검색)에서 후보를 몇 개 가져올지
DEFAULT_SEED_K = 3

# 문맥 확장 폭: seed turn_index 기준 ±window
DEFAULT_WINDOW = 1

# seed에서 가장 좋은 문서의 session_id만 사용 (상담 도메인에 더 적합)
DEFAULT_USE_BEST_SESSION_ONLY = True


# -------------------------------------------------------------
# VectorDB Load
# -------------------------------------------------------------
def load_vector_db_for_contextual(persist_directory: Optional[str] = None) -> VectorStore:
    """
    Contextual Retriever용 VectorStore 로드
    - retriever.py의 load_vector_db를 사용해 문서가 있는 컬렉션 자동 선택
    """
    return load_vector_db(persist_directory=persist_directory)


# -------------------------------------------------------------
# Internal Helpers
# -------------------------------------------------------------
def _normalize_where(where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    where가 비어있으면 None 반환
    """
    if not where:
        return None
    return where


def _extract_session_turn(meta: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    """
    metadata에서 session_id, turn_index 추출
    """
    if not meta:
        return None, None
    session_id = meta.get("session_id")
    turn_index = meta.get("turn_index")
    if isinstance(turn_index, bool):  # 혹시 True/False로 들어오는 이상 케이스 방지
        return session_id, None
    try:
        turn_index = int(turn_index) if turn_index is not None else None
    except Exception:
        turn_index = None
    return session_id, turn_index


def _get_doc_by_session_turn(
    vector_db: VectorStore,
    session_id: str,
    turn_index: int
) -> Optional[Tuple[str, Dict[str, Any], str]]:
    """
    특정 (session_id, turn_index)에 해당하는 문서를 Chroma에서 직접 조회합니다.
    - VectorStore 래퍼에 없으므로 vector_db.collection.get(where=...) 사용
    Returns: (doc_text, metadata, doc_id) or None
    """
    try:
        got = vector_db.collection.get(
            where={"session_id": session_id, "turn_index": turn_index},
            include=["documents", "metadatas"]
        )
    except Exception:
        return None

    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    ids = got.get("ids") or []

    if not docs:
        return None

    doc_text = docs[0]
    meta = metas[0] if metas else {}
    doc_id = ids[0] if ids else f"{session_id}__turn_{turn_index}"
    return doc_text, meta, doc_id


def _distance_to_query(
    vector_db: VectorStore,
    query: str,
    session_id: str,
    turn_index: int
) -> Optional[float]:
    """
    특정 (session_id, turn_index) 문서에 대해, query와의 distance를 Chroma query로 계산해 반환.
    - where로 문서를 강제하고 n_results=1로 거리만 얻습니다.
    """
    try:
        r = vector_db.collection.query(
            query_texts=[query],
            n_results=1,
            where={"session_id": session_id, "turn_index": turn_index}
        )
        dists = (r.get("distances") or [[]])[0]
        if dists:
            return float(dists[0])
    except Exception:
        pass
    return None


# -------------------------------------------------------------
# Retriever Factory
# -------------------------------------------------------------
def create_contextual_retriever(
    vector_db: VectorStore,
    top_k: int = DEFAULT_TOP_K,
    seed_k: int = DEFAULT_SEED_K,
    window: int = DEFAULT_WINDOW,
    use_best_session_only: bool = DEFAULT_USE_BEST_SESSION_ONLY,
):
    """
    Contextual Retriever 생성 (함수 기반)

    동작:
      1) similarity 기반 seed 검색 (seed_k)
      2) seed들의 session_id/turn_index 확인
      3) (best seed 기준) 같은 session_id에서 turn_index ± window 확장
      4) 결과를 turn_index 순으로 정렬 후 top_k로 자르거나(기본) 전부 반환(선택 구현 가능)

    반환 포맷:
      [{"content": str, "metadata": dict, "distance": float}, ...]
    """

    top_k = int(top_k)
    seed_k = int(seed_k)
    window = int(window)

    def retriever(
        query: str,
        category: Optional[str] = None,
        speaker: Optional[str] = None,
        min_severity: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        # -----------------------------
        # metadata 필터 구성 (seed 검색용)
        # -----------------------------
        where: Dict[str, Any] = {}

        # 프로젝트 내 기존 필터 키들과 실제 메타 키가 다를 수 있으므로
        # 사장님 메타에서 존재하는 키만 사용 권장
        # (예: category_paragraph_speaker, default_category 등)
        if category:
            # 사장님 메타 출력에는 default_category가 있었음
            # category가 default_category 기준이면 아래 유지
            where["default_category"] = category

        if speaker:
            # 사장님 메타 출력에는 category_paragraph_speaker가 있었음
            where["category_paragraph_speaker"] = speaker

        if min_severity is not None:
            # 사장님 메타에 severity가 없을 수 있음(출력엔 보이지 않았음)
            # 있는 경우에만 필터가 의미가 있으니, 여기선 조건만 추가해 둠
            where["severity"] = {"$gte": min_severity}

        where = _normalize_where(where)

        # -----------------------------
        # 1) Seed 검색 (Similarity)
        # -----------------------------
        base = vector_db.search(
            query=query,
            n_results=seed_k,
            where=where
        )

        seed_docs = base.get("documents", []) or []
        seed_metas = base.get("metadatas", []) or []
        seed_dists = base.get("distances", []) or []
        seed_ids = base.get("ids", []) or []

        if not seed_docs:
            return []

        # seed에서 session/turn 추출
        seeds: List[Dict[str, Any]] = []
        for i in range(len(seed_docs)):
            meta = seed_metas[i] if i < len(seed_metas) else {}
            sid, tid = _extract_session_turn(meta)
            dist = seed_dists[i] if i < len(seed_dists) else None
            doc_id = seed_ids[i] if i < len(seed_ids) else None
            seeds.append({
                "content": seed_docs[i],
                "metadata": meta,
                "distance": dist,
                "session_id": sid,
                "turn_index": tid,
                "id": doc_id
            })

        # session_id/turn_index 없는 seed는 제외
        seeds = [s for s in seeds if s["session_id"] is not None and s["turn_index"] is not None]
        if not seeds:
            # 메타에 session/turn이 없는 데이터셋이면 contextual 불가
            # (사장님 데이터는 존재함)
            return [{
                "content": seed_docs[0],
                "metadata": seed_metas[0] if seed_metas else {},
                "distance": seed_dists[0] if seed_dists else None,
            }]

        # -----------------------------
        # 2) 사용할 session_id 결정
        # -----------------------------
        # best seed: distance가 가장 작은 것
        def _dist_val(x):
            return float(x["distance"]) if x["distance"] is not None else 1e9

        seeds_sorted = sorted(seeds, key=_dist_val)
        best_session_id = seeds_sorted[0]["session_id"]

        target_sessions = {best_session_id} if use_best_session_only else set(s["session_id"] for s in seeds_sorted)

        # -----------------------------
        # 3) turn_index window 확장 (세션별)
        # -----------------------------
        collected: Dict[str, Dict[str, Any]] = {}  # id -> item
        visited_turns: Set[Tuple[str, int]] = set()

        for sid in target_sessions:
            # 해당 세션의 seed turn만 모아서 확장
            seed_turns = sorted({s["turn_index"] for s in seeds_sorted if s["session_id"] == sid})

            # seed turn들 각각에 대해 window 확장
            for t in seed_turns:
                for tt in range(t - window, t + window + 1):
                    if tt < 0:
                        continue
                    key = (sid, tt)
                    if key in visited_turns:
                        continue
                    visited_turns.add(key)

                    got = _get_doc_by_session_turn(vector_db, sid, tt)
                    if not got:
                        continue

                    doc_text, meta, doc_id = got

                    # query 대비 distance 계산 (통일성 유지)
                    dist = _distance_to_query(vector_db, query, sid, tt)

                    collected[doc_id] = {
                        "content": doc_text,
                        "metadata": meta,
                        "distance": dist,
                        "_session_id": sid,
                        "_turn_index": tt
                    }

        if not collected:
            # window 내 문서를 못 찾는 예외 케이스: seed만 반환
            out = []
            for s in seeds_sorted[:top_k]:
                out.append({"content": s["content"], "metadata": s["metadata"], "distance": s["distance"]})
            return out

        # -----------------------------
        # 4) 정렬/절단
        #   - 상담 문맥은 turn 순서가 중요하므로 turn_index 오름차순 정렬
        #   - 여러 세션이면 session_id로 묶고 turn 정렬
        # -----------------------------
        items = list(collected.values())
        items.sort(key=lambda x: (x.get("_session_id", ""), x.get("_turn_index", 0)))

        # top_k는 "문맥 조각 개수"로 이해 (window가 커지면 증가)
        # 실험 비교를 위해 기본은 top_k로 제한
        if top_k > 0 and len(items) > top_k:
            # 문맥 유지가 깨지지 않도록: 거리 기반으로 자르면 흐름이 찢어질 수 있음
            # 따라서 "best seed turn" 중심으로 가까운 turn을 우선 채택
            # 1) best seed turn을 기준으로 turn 거리로 정렬 → top_k 선택 → 다시 turn 순으로 정렬
            best_turn = seeds_sorted[0]["turn_index"]

            def _turn_prox(it):
                return abs(int(it.get("_turn_index", 0)) - int(best_turn))

            chosen = sorted(items, key=_turn_prox)[:top_k]
            chosen.sort(key=lambda x: (x.get("_session_id", ""), x.get("_turn_index", 0)))
            items = chosen

        # 최종 포맷팅 (내부키 제거)
        formatted: List[Dict[str, Any]] = []
        for it in items:
            formatted.append({
                "content": it["content"],
                "metadata": it["metadata"],
                "distance": it["distance"],
            })

        return formatted

    print("[INFO] Contextual Retriever created")
    print(f"       top_k                = {top_k}")
    print(f"       seed_k               = {seed_k}")
    print(f"       window               = {window}")
    print(f"       use_best_session_only= {use_best_session_only}")

    return retriever


# -------------------------------------------------------------
# Debug Functions
# -------------------------------------------------------------
def debug_retriever(retriever, query: str):
    print("\n[DEBUG] Retriever Test (Contextual)")
    print(f"[DEBUG] Query: {query}")

    start_time = time.time()
    results = retriever(query)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"[DEBUG] Retrieved documents count: {len(results)}")
    print(f"[TIME]: {elapsed_time:.4f}초")
    
    for idx, r in enumerate(results[:5]):
        meta = r.get("metadata") or {}
        sid = meta.get("session_id")
        tid = meta.get("turn_index")
        spk = meta.get("category_paragraph_speaker")
        print(f"\n[DEBUG] Doc {idx+1} | session={sid} turn={tid} speaker={spk}")
        print("-" * 60)
        print((r.get("content") or "")[:300])
        print("[DIST]", r.get("distance"))


# -------------------------------------------------------------
# Main (Standalone Test)
# -------------------------------------------------------------
def main():
    start_total = time.time()
    
    vector_db = load_vector_db_for_contextual()
    
    load_end = time.time()
    print(f"[TIME] VectorDB 로딩: {load_end - start_total:.4f}초")

    retriever = create_contextual_retriever(
        vector_db=vector_db,
        top_k=DEFAULT_TOP_K,
        seed_k=DEFAULT_SEED_K,
        window=DEFAULT_WINDOW,
        use_best_session_only=DEFAULT_USE_BEST_SESSION_ONLY
    )

    test_queries = [
        "요즘 너무 불안해서 잠이 안 와",
        "계속 실패하는 느낌이야",
        "아무것도 하기 싫어",
        "미래가 걱정돼",
        "위로가 필요해"
    ]

    for q in test_queries:
        debug_retriever(retriever, q)
    
    end_total = time.time()
    print(f"\n[TIME] 전체 실행 시간: {end_total - start_total:.4f}초")


# -------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
