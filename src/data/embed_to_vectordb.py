"""
FileName    : embed_to_vectordb.py
Auth        : 박수빈
Date        : 2026-01-06
Description : 전처리된 JSONL 데이터를 ChromaDB에 임베딩하여 저장
Issue/Note  : docs_for_vectordb.jsonl → ChromaDB 벡터 스토어
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database import VectorStore

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

DEFAULT_INPUT_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "docs_for_vectordb.jsonl"
BATCH_SIZE = 100  # 한 번에 저장할 문서 수


# -------------------------------------------------------------
# Metadata Flatten Function
# -------------------------------------------------------------

def flatten_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    중첩된 metadata를 ChromaDB에 저장 가능한 형태로 평탄화
    
    ChromaDB는 메타데이터 값으로 str, int, float, bool만 지원
    """
    flat = {}
    
    for key, value in metadata.items():
        if isinstance(value, dict):
            # 중첩된 dict는 주요 필드만 추출
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (str, int, float, bool)):
                    flat[f"{key}_{sub_key}"] = sub_value
        elif isinstance(value, (str, int, float, bool)):
            flat[key] = value
        elif value is None:
            flat[key] = ""  # None → 빈 문자열로 변환
        else:
            flat[key] = str(value)  # 기타 → 문자열로 변환
    
    return flat


# -------------------------------------------------------------
# Load and Embed Functions
# -------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """JSONL 파일 로드"""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def embed_documents(
    input_path: Path = DEFAULT_INPUT_PATH,
    batch_size: int = BATCH_SIZE,
    limit: int = None
) -> Dict[str, int]:
    """
    JSONL 파일의 문서들을 ChromaDB에 임베딩
    
    Args:
        input_path: 입력 JSONL 파일 경로
        batch_size: 배치 크기
        limit: 임베딩할 최대 문서 수 (테스트용)
    
    Returns:
        {'total': 368944, 'embedded': 368944, 'skipped': 0}
    """
    print(f"📂 입력 파일: {input_path}")
    
    # 1. 데이터 로드
    print("📖 데이터 로드 중...")
    docs = load_jsonl(input_path)
    total = len(docs)
    
    if limit:
        docs = docs[:limit]
        print(f"   (테스트 모드: {limit}개만 처리)")
    
    print(f"   전체 문서 수: {total:,}")
    
    # 2. VectorStore 초기화
    print("\n🔧 VectorStore 초기화...")
    vs = VectorStore()
    initial_count = vs.get_document_count()
    print(f"   현재 저장된 문서: {initial_count:,}")
    
    # 3. 배치 단위로 임베딩
    print(f"\n🚀 임베딩 시작 (배치 크기: {batch_size})...")
    
    stats = {'total': len(docs), 'embedded': 0, 'skipped': 0}
    
    for i in tqdm(range(0, len(docs), batch_size), desc="임베딩 진행"):
        batch = docs[i:i + batch_size]
        
        # 문서 텍스트, 메타데이터, ID 추출
        texts = []
        metadatas = []
        doc_ids = []
        
        for idx, doc in enumerate(batch):
            global_idx = i + idx
            texts.append(doc['text'])
            
            # 메타데이터 평탄화
            flat_meta = flatten_metadata(doc.get('metadata', {}))
            metadatas.append(flat_meta)
            
            # 고유 ID 생성 (session_id + turn_index)
            session_id = flat_meta.get('session_id', f'doc_{global_idx}')
            turn_index = flat_meta.get('turn_index', global_idx)
            doc_id = f"{session_id}_turn_{turn_index}"
            doc_ids.append(doc_id)
        
        # VectorStore에 추가
        try:
            new_ids = vs.add_documents(
                documents=texts, 
                metadatas=metadatas,
                ids=doc_ids
            )
            stats['embedded'] += len(new_ids)
            stats['skipped'] += len(batch) - len(new_ids)
        except Exception as e:
            print(f"\n⚠️ 배치 {i//batch_size + 1} 에러: {e}")
            stats['skipped'] += len(batch)
    
    # 4. 결과 출력
    final_count = vs.get_document_count()
    print(f"\n✅ 임베딩 완료!")
    print(f"   - 처리 요청: {stats['total']:,}")
    print(f"   - 새로 추가: {stats['embedded']:,}")
    print(f"   - 스킵 (중복): {stats['skipped']:,}")
    print(f"   - 최종 문서 수: {final_count:,}")
    
    return stats


# -------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSONL 데이터를 VectorDB에 임베딩")
    parser.add_argument(
        "--input", 
        type=str, 
        default=str(DEFAULT_INPUT_PATH),
        help="입력 JSONL 파일 경로"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=BATCH_SIZE,
        help="배치 크기 (기본: 100)"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="테스트용 최대 문서 수"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)
    
    embed_documents(
        input_path=input_path,
        batch_size=args.batch_size,
        limit=args.limit
    )
