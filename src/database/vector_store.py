"""
FileName    : vector_store.py
Auth        : 박수빈, 우재현
Date        : 2026-01-28
Description : ChromaDB 벡터 스토어 래퍼 클래스 - 상담 데이터 임베딩 및 유사도 검색
Issue/Note  : OpenAI Embedding 사용, 메타데이터 필터링 지원, Pydantic Settings 적용
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import hashlib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.db_config import db_settings
from config.model_config import model_settings

# -------------------------------------------------------------
# Vector Store Class
# -------------------------------------------------------------

class VectorStore:
    """
    ChromaDB 기반 벡터 스토어 관리 클래스
    
    주요 기능:
    - 상담 단락 임베딩 저장
    - 유사도 검색 (의미 기반)
    - 메타데이터 필터링 (카테고리, 증상 등)
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        VectorStore 초기화
        
        Args:
            persist_directory: ChromaDB 저장 경로 (None이면 기본 경로 사용)
        """
        # 디렉토리 생성
        db_settings.ensure_directories()
        
        # ChromaDB 클라이언트 초기화
        persist_path = persist_directory or str(db_settings.CHROMA_DB_DIR)
        self.client = chromadb.PersistentClient(path=persist_path)
        
        # 컬렉션 가져오기 또는 생성
        self.collection = self.client.get_or_create_collection(
            name=db_settings.CHROMA_COLLECTION_NAME,
            metadata={"description": "심리 상담 데이터 임베딩 컬렉션"}
        )
    
    # -------------------------------------------------------------
    # Document Management
    # -------------------------------------------------------------
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        문서들을 벡터 스토어에 추가
        
        Args:
            documents: 문서 텍스트 리스트
            metadatas: 각 문서의 메타데이터 리스트
            ids: 문서 ID 리스트 (None이면 자동 생성)
        
        Returns:
            저장된 문서 ID 리스트
        """
        if ids is None:
            # 문서 내용 기반으로 고유 ID 생성
            ids = [self._generate_id(doc, meta) for doc, meta in zip(documents, metadatas)]
        
        # 이미 존재하는 ID 필터링
        existing_ids = set(self.collection.get(ids=ids)["ids"])
        
        new_docs = []
        new_metas = []
        new_ids = []
        
        for doc, meta, doc_id in zip(documents, metadatas, ids):
            if doc_id not in existing_ids:
                new_docs.append(doc)
                new_metas.append(meta)
                new_ids.append(doc_id)
        
        if new_docs:
            self.collection.add(
                documents=new_docs,
                metadatas=new_metas,
                ids=new_ids
            )
        
        return new_ids
    
    def _generate_id(self, document: str, metadata: Dict[str, Any]) -> str:
        """
        문서와 메타데이터를 기반으로 고유 ID 생성
        """
        content = f"{metadata.get('counseling_id', '')}_{metadata.get('paragraph_index', '')}_{document[:50]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    # -------------------------------------------------------------
    # Search Methods
    # -------------------------------------------------------------
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        유사도 검색 수행
        
        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 수
            where: 메타데이터 필터 조건
            where_document: 문서 내용 필터 조건
        
        Returns:
            검색 결과 (ids, documents, metadatas, distances)
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }
    
    def search_by_category(
        self,
        query: str,
        category: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        카테고리별 유사도 검색
        
        Args:
            query: 검색 쿼리
            category: 카테고리 (DEPRESSION, ANXIETY, ADDICTION, NORMAL)
            n_results: 반환할 결과 수
        """
        return self.search(
            query=query,
            n_results=n_results,
            where={"category": category}
        )
    
    def search_by_speaker(
        self,
        query: str,
        speaker: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        화자별 유사도 검색
        
        Args:
            query: 검색 쿼리
            speaker: 화자 (상담사, 내담자)
            n_results: 반환할 결과 수
        """
        return self.search(
            query=query,
            n_results=n_results,
            where={"speaker": speaker}
        )
    
    def search_high_severity(
        self,
        query: str,
        min_severity: int = 2,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        고위험 케이스 검색 (severity >= min_severity)
        
        Args:
            query: 검색 쿼리
            min_severity: 최소 심각도 (0-3)
            n_results: 반환할 결과 수
        """
        return self.search(
            query=query,
            n_results=n_results,
            where={"severity": {"$gte": min_severity}}
        )
    
    # -------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------
    
    def get_document_count(self) -> int:
        """
        저장된 문서 수 반환
        """
        return self.collection.count()
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        문서 삭제
        
        Args:
            ids: 삭제할 문서 ID 리스트
        """
        self.collection.delete(ids=ids)
    
    def clear_collection(self) -> None:
        """
        컬렉션의 모든 문서 삭제 (주의: 복구 불가)
        """
        # 컬렉션 삭제 후 재생성
        self.client.delete_collection(db_settings.CHROMA_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=db_settings.CHROMA_COLLECTION_NAME,
            metadata={"description": "심리 상담 데이터 임베딩 컬렉션"}
        )
    
    def get_all_documents(self, limit: int = 100) -> Dict[str, Any]:
        """
        모든 문서 조회 (테스트/디버깅용)
        
        Args:
            limit: 최대 반환 문서 수
        """
        return self.collection.get(limit=limit)


# -------------------------------------------------------------
# Entry Point (테스트용)
# -------------------------------------------------------------

if __name__ == "__main__":
    print("VectorStore 초기화 테스트...")
    
    # VectorStore 인스턴스 생성
    vector_store = VectorStore()
    print(f"컬렉션 생성 완료: {db_settings.CHROMA_COLLECTION_NAME}")
    print(f"현재 문서 수: {vector_store.get_document_count()}")

    
    # 테스트 문서 추가
    test_docs = [
        "요즘 너무 우울하고 아무것도 하기 싫어요.",
        "잠을 잘 못 자고 불안한 느낌이 계속 들어요.",
        "일상생활에서 스트레스를 많이 받고 있습니다."
    ]
    test_metas = [
        {"counseling_id": 1, "paragraph_index": 0, "speaker": "내담자", "category": "DEPRESSION", "severity": 2},
        {"counseling_id": 2, "paragraph_index": 0, "speaker": "내담자", "category": "ANXIETY", "severity": 1},
        {"counseling_id": 3, "paragraph_index": 0, "speaker": "내담자", "category": "NORMAL", "severity": 0}
    ]
    
    ids = vector_store.add_documents(test_docs, test_metas)
    print(f"테스트 문서 추가 완료: {len(ids)}개")
    
    # 검색 테스트
    results = vector_store.search("우울한 기분이 들어요", n_results=2)
    print(f"\n검색 결과:")
    for doc, meta in zip(results["documents"], results["metadatas"]):
        print(f"  - [{meta['category']}] {doc[:50]}...")
    
    print(f"\n최종 문서 수: {vector_store.get_document_count()}")
