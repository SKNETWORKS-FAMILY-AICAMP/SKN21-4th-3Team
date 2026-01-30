"""
FileName    : evaluate_rag.py
Auth        : 박수빈
Date        : 2026-01-28
Description : RAG 시스템 정량적 평가 (Recall@k)
Issue/Note  : 평가서 피드백 반영 - 정량적 성능 측정 추가
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import VectorStore


# -------------------------------------------------------------
# Evaluation Functions
# -------------------------------------------------------------

def load_test_queries(path: Path) -> List[Dict[str, Any]]:
    """
    테스트 쿼리 로드
    
    Args:
        path: 테스트 쿼리 JSON 파일 경로
    
    Returns:
        [{"query": "...", "expected_session_ids": [...], "category": "..."}]
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_recall_at_k(
    test_queries: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5, 10],
    verbose: bool = False
) -> Dict[str, float]:
    """
    Recall@k 평가 실행
    
    Args:
        test_queries: 테스트 쿼리 리스트
        k_values: 평가할 k 값 리스트
        verbose: 상세 출력 여부
    
    Returns:
        {"recall@1": 0.65, "recall@3": 0.82, "recall@5": 0.91, ...}
    """
    vs = VectorStore()
    
    results = {f"recall@{k}": 0.0 for k in k_values}
    hits = {k: 0 for k in k_values}
    
    for i, tq in enumerate(test_queries):
        query = tq["query"]
        expected_ids = set(tq.get("expected_session_ids", []))
        
        if not expected_ids:
            continue
        
        # 가장 큰 k 값으로 검색
        max_k = max(k_values)
        search_results = vs.search(query, n_results=max_k)
        
        # 검색된 session_id 추출
        retrieved_ids = []
        metadatas = search_results.get("metadatas", [])
        for meta in metadatas:
            session_id = meta.get("session_id", "") if isinstance(meta, dict) else ""
            if session_id:
                retrieved_ids.append(session_id)
        
        if verbose:
            print(f"\n[{i+1}] Query: {query}")
            print(f"    Expected: {expected_ids}")
            print(f"    Retrieved: {retrieved_ids[:5]}...")
        
        # 각 k 값에 대해 hit 체크
        for k in k_values:
            top_k = set(retrieved_ids[:k])
            if expected_ids & top_k:  # intersection이 있으면 hit
                hits[k] += 1
    
    # Recall 계산
    total = len([tq for tq in test_queries if tq.get("expected_session_ids")])
    for k in k_values:
        results[f"recall@{k}"] = hits[k] / total if total > 0 else 0.0
    
    return results


def evaluate_mrr(
    test_queries: List[Dict[str, Any]],
    max_k: int = 10,
    verbose: bool = False
) -> float:
    """
    MRR (Mean Reciprocal Rank) 평가
    
    Args:
        test_queries: 테스트 쿼리 리스트
        max_k: 최대 검색 결과 수
        verbose: 상세 출력 여부
    
    Returns:
        MRR 점수 (0.0 ~ 1.0)
    """
    vs = VectorStore()
    
    reciprocal_ranks = []
    
    for i, tq in enumerate(test_queries):
        query = tq["query"]
        expected_ids = set(tq.get("expected_session_ids", []))
        
        if not expected_ids:
            continue
        
        search_results = vs.search(query, n_results=max_k)
        
        # 첫 번째 hit의 rank 찾기
        rank = 0
        metadatas = search_results.get("metadatas", [])
        for j, meta in enumerate(metadatas):
            session_id = meta.get("session_id", "") if isinstance(meta, dict) else ""
            if session_id in expected_ids:
                rank = j + 1
                break
        
        if rank > 0:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
        
        if verbose:
            print(f"[{i+1}] Query: {query}, Rank: {rank if rank > 0 else 'Not found'}")
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def print_evaluation_report(
    recall_results: Dict[str, float],
    mrr: float,
    total_queries: int
) -> None:
    """
    평가 결과 리포트 출력
    """
    print("\n" + "=" * 60)
    print("📊 RAG 평가 결과")
    print("=" * 60)
    print(f"\n총 테스트 쿼리 수: {total_queries}")
    
    print("\n[Recall@k]")
    for metric, value in sorted(recall_results.items()):
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        print(f"   {metric}: {value:.2%} |{bar}|")
    
    print(f"\n[MRR (Mean Reciprocal Rank)]")
    bar = "█" * int(mrr * 20) + "░" * (20 - int(mrr * 20))
    print(f"   MRR: {mrr:.4f} |{bar}|")
    
    print("\n" + "=" * 60)


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main(
    test_queries_path: str,
    k_values: List[int] = [1, 3, 5, 10],
    verbose: bool = False,
    output_path: str = None
) -> Dict[str, Any]:
    """
    RAG 평가 메인 함수
    """
    path = Path(test_queries_path)
    
    if not path.exists():
        print(f"❌ 테스트 쿼리 파일 없음: {path}")
        return {}
    
    print(f"📂 테스트 쿼리 로드: {path}")
    test_queries = load_test_queries(path)
    print(f"   쿼리 수: {len(test_queries)}")
    
    # Recall@k 평가
    print("\n🔍 Recall@k 평가 중...")
    recall_results = evaluate_recall_at_k(test_queries, k_values, verbose)
    
    # MRR 평가
    print("🔍 MRR 평가 중...")
    mrr = evaluate_mrr(test_queries, max_k=max(k_values), verbose=verbose)
    
    # 결과 출력
    print_evaluation_report(recall_results, mrr, len(test_queries))
    
    # 결과 저장
    results = {
        "total_queries": len(test_queries),
        "recall": recall_results,
        "mrr": mrr
    }
    
    if output_path:
        output = Path(output_path)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 결과 저장: {output}")
    
    return results


# -------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 시스템 정량적 평가")
    parser.add_argument(
        "--test_queries",
        type=str,
        default="tests/test_queries.json",
        help="테스트 쿼리 JSON 파일 경로"
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="평가할 k 값들"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 출력"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 저장 경로"
    )
    
    args = parser.parse_args()
    
    main(
        test_queries_path=args.test_queries,
        k_values=args.k_values,
        verbose=args.verbose,
        output_path=args.output
    )
