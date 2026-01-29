"""
테스트 쿼리에 대한 검색 결과 확인 및 라벨링 도우미
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import VectorStore

def search_and_show(query: str, n_results: int = 5):
    """각 쿼리에 대해 상위 결과 반환"""
    vs = VectorStore()
    results = vs.search(query, n_results=n_results)
    
    session_ids = []
    print(f"\n🔍 Query: '{query}'")
    print("-" * 50)
    
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    distances = results.get("distances", [])
    
    for i in range(len(documents)):
        meta = metadatas[i] if i < len(metadatas) else {}
        session_id = meta.get("session_id", "N/A")
        category = meta.get("category", "N/A")
        content = documents[i][:80] if i < len(documents) else ""
        distance = distances[i] if i < len(distances) else 0
        
        session_ids.append(session_id)
        print(f"  [{i+1}] session_id: {session_id}")
        print(f"      category: {category}, distance: {distance:.4f}")
        print(f"      content: {content}...")
    
    return session_ids

# 테스트 쿼리 로드
test_queries_path = PROJECT_ROOT / "tests" / "test_queries.json"
with open(test_queries_path, "r", encoding="utf-8") as f:
    test_queries = json.load(f)

# 각 쿼리에 대해 검색 결과 확인
all_results = {}
for tq in test_queries:
    query = tq["query"]
    session_ids = search_and_show(query, n_results=3)
    all_results[query] = list(set(session_ids))  # 중복 제거

print("\n\n" + "=" * 60)
print("📝 라벨링 결과 (expected_session_ids)")
print("=" * 60)
print(json.dumps(all_results, ensure_ascii=False, indent=2))
