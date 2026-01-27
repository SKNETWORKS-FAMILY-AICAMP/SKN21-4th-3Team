"""
FileName    : vector_loader.py
Auth        : 박수빈
Date        : 2026-01-05
Description : Vector DB 공통 함수 (저장 + 조회)
Issue/Note  : Save: 전처리된 데이터 저장 / Load: 검색, 전체조회, ID조회
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database import DatabaseManager

# -------------------------------------------------------------
# Vector DB Load Functions
# -------------------------------------------------------------

def load_counseling_to_db(
    data: Dict[str, Any],
    db: Optional[DatabaseManager] = None
) -> Dict[str, int]:
    """
    단일 상담 데이터를 DB에 저장 (SQLite + VectorDB)
    
    Args:
        data: 전처리된 상담 데이터
            {
                'source_id': 'D012',
                'category': 'DEPRESSION',
                'metadata': {...},
                'paragraphs': [...],
                'labels': [...]
            }
        db: DatabaseManager 인스턴스 (None이면 새로 생성)
    
    Returns:
        {'counseling_id': 1, 'paragraphs_saved': 537}
    """
    close_db = False
    if db is None:
        db = DatabaseManager()
        close_db = True
    
    try:
        # 1. counseling_data 저장
        metadata = data.get('metadata', {})
        severity = _calculate_severity(metadata)
        
        counseling = db.add_counseling_data(
            source_id=data['source_id'],
            category=data.get('category', metadata.get('class', 'UNKNOWN')),
            summary=metadata.get('summary'),
            severity=severity,
            source_file=data.get('txt_path'),
            has_detailed_label=bool(data.get('labels')),
            raw_metadata=metadata
        )
        
        # 2. paragraphs 저장 (VectorDB 포함)
        paragraphs = data.get('paragraphs', [])
        labels_list = data.get('labels', [])
        paragraphs_saved = 0
        
        for idx, para in enumerate(paragraphs):
            labels = labels_list[idx] if idx < len(labels_list) else None
            
            db.add_counseling_paragraph(
                counseling_id=counseling.id,
                paragraph_index=para.get('index', idx),
                speaker=para.get('speaker', 'UNKNOWN'),
                content=para.get('content', ''),
                labels=labels,
                category=data.get('category'),
                severity=severity
            )
            paragraphs_saved += 1
        
        return {
            'counseling_id': counseling.id,
            'paragraphs_saved': paragraphs_saved
        }
    
    finally:
        if close_db:
            db.close()


def load_batch_to_db(
    data_list: List[Dict[str, Any]],
    db: Optional[DatabaseManager] = None
) -> Dict[str, int]:
    """
    여러 상담 데이터를 일괄 저장
    
    Args:
        data_list: 전처리된 상담 데이터 리스트
        db: DatabaseManager 인스턴스
    
    Returns:
        {
            'total': 50,
            'success': 48,
            'paragraphs': 15000,
            'error': 2
        }
    """
    close_db = False
    if db is None:
        db = DatabaseManager()
        close_db = True
    
    stats = {
        'total': len(data_list),
        'success': 0,
        'paragraphs': 0,
        'error': 0
    }
    
    try:
        for data in data_list:
            try:
                result = load_counseling_to_db(data, db)
                stats['success'] += 1
                stats['paragraphs'] += result['paragraphs_saved']
                print(f"✓ {data.get('source_id')} 저장 완료 ({result['paragraphs_saved']}개 단락)")
            except Exception as e:
                stats['error'] += 1
                print(f"✗ {data.get('source_id')} 에러: {e}")
        
        return stats
    
    finally:
        if close_db:
            db.close()


def _calculate_severity(metadata: Dict) -> int:
    """심각도 계산 (0~3)"""
    d = metadata.get('depression', 0) or 0
    a = metadata.get('anxiety', 0) or 0
    add = metadata.get('addiction', 0) or 0
    
    total = d + a + add
    if total >= 2:
        return 3  # 중증
    elif total >= 1:
        return 2  # 중등
    else:
        return 0  # 정상


# -------------------------------------------------------------
# Vector DB Load (조회) Functions
# -------------------------------------------------------------

def search_similar(
    query: str,
    n_results: int = 5,
    category: Optional[str] = None,
    speaker: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    유사 상담 사례 검색
    
    Args:
        query: 검색 쿼리 (예: "요즘 우울해요")
        n_results: 반환할 결과 수
        category: 카테고리 필터 (DEPRESSION, ANXIETY, ADDICTION, NORMAL)
        speaker: 화자 필터 (상담사, 내담자)
    
    Returns:
        [
            {
                'content': '발화 내용',
                'speaker': '내담자',
                'category': 'DEPRESSION',
                'distance': 0.2
            },
            ...
        ]
    """
    from src.database import VectorStore
    vs = VectorStore()
    
    # 필터 조건 구성
    where = {}
    if category:
        where['category'] = category
    if speaker:
        where['speaker'] = speaker
    
    results = vs.search(
        query=query,
        n_results=n_results,
        where=where if where else None
    )
    
    # 결과 포맷팅
    formatted = []
    for i, doc in enumerate(results['documents']):
        formatted.append({
            'content': doc,
            'metadata': results['metadatas'][i] if results['metadatas'] else {},
            'distance': results['distances'][i] if results['distances'] else None
        })
    
    return formatted


def get_all_documents(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Vector DB의 모든 문서 조회
    
    Args:
        limit: 최대 반환 문서 수
    
    Returns:
        [
            {'id': 'abc123', 'content': '...', 'metadata': {...}},
            ...
        ]
    """
    from src.database import VectorStore
    vs = VectorStore()
    
    results = vs.get_all_documents(limit=limit)
    
    formatted = []
    for i, doc_id in enumerate(results['ids']):
        formatted.append({
            'id': doc_id,
            'content': results['documents'][i] if results['documents'] else None,
            'metadata': results['metadatas'][i] if results['metadatas'] else {}
        })
    
    return formatted


def get_by_ids(ids: List[str]) -> List[Dict[str, Any]]:
    """
    ID로 문서 조회
    
    Args:
        ids: 조회할 문서 ID 리스트
    
    Returns:
        [
            {'id': 'abc123', 'content': '...', 'metadata': {...}},
            ...
        ]
    """
    from src.database import VectorStore
    vs = VectorStore()
    
    results = vs.collection.get(ids=ids)
    
    formatted = []
    for i, doc_id in enumerate(results['ids']):
        formatted.append({
            'id': doc_id,
            'content': results['documents'][i] if results['documents'] else None,
            'metadata': results['metadatas'][i] if results['metadatas'] else {}
        })
    
    return formatted


def get_document_count() -> int:
    """Vector DB 문서 수 조회"""
    from src.database import VectorStore
    vs = VectorStore()
    return vs.get_document_count()


# -------------------------------------------------------------
# Entry Point (테스트용)
# -------------------------------------------------------------

if __name__ == "__main__":
    print("=== Vector DB 함수 테스트 ===\n")
    
    # 문서 수 확인
    count = get_document_count()
    print(f"1. 현재 문서 수: {count}")
    
    # 전체 조회 (5개만)
    docs = get_all_documents(limit=5)
    print(f"\n2. 전체 조회 (상위 5개):")
    for d in docs[:3]:
        print(f"   - {d['content'][:40]}...")
    
    # 유사 검색
    if count > 0:
        results = search_similar("우울해요", n_results=3)
        print(f"\n3. '우울해요' 검색 결과:")
        for r in results:
            print(f"   - {r['content'][:40]}...")

