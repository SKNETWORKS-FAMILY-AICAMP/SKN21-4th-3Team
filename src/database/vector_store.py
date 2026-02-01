"""
FileName    : vector_store.py
Auth        : 박수빈, 우재현
Date        : 2026-01-30
Description : ChromaDB 기반 벡터 스토어 클래스 - 상담 데이터 임베딩 및 유사도 검색
Issue/Note  : RDS 마이그레이션 취소 -> ChromaDB 복구
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

from typing import List, Dict, Any, Optional
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.model_config import model_settings

# -------------------------------------------------------------
# Vector Store Class
# -------------------------------------------------------------

class VectorStore:
    """
    ChromaDB 기반 벡터 스토어 관리 클래스
    
    주요 기능:
    - 상담 단락 임베딩 저장 (Local ChromaDB)
    - 유사도 검색 (L2 Distance)
    """
    
    def __init__(self):
        """
        VectorStore 초기화
        """
        # 데이터 경로 설정 (프로젝트 루트 / data / vector_store)
        self.persist_directory = os.path.join(os.getcwd(), "data", "vector_store")
        
        # 디렉토리 생성
        os.makedirs(self.persist_directory, exist_ok=True)
        
        print(f"[VectorStore] Initializing ChromaDB at: {self.persist_directory}")
        
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # 임베딩 모델 초기화 (SentenceTransformer)
        try:
            from sentence_transformers import SentenceTransformer
            # 모델 설정 (config/model_config.py에 없는 경우 대비 하드코딩)
            model_name = "jhgan/ko-sroberta-multitask" 
            device = "cpu"
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            
            print(f"[VectorStore] Loading embedding model: {model_name} on {device}")
            self.embedding_model = SentenceTransformer(
                model_name, 
                device=device
            )
            print(f"[VectorStore] Model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load embedding model: {e}")
            raise e
            
        # 컬렉션 가져오기 또는 생성
        self.collection_name = "psych_counseling_vectors"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"} # 코사인 유사도 사용
        )
            
    def get_document_count(self) -> int:
        """저장된 문서 수 반환"""
        return self.collection.count()

    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None):
        """
        문서 및 임베딩 저장
        
        Args:
            documents: 저장할 텍스트 리스트
            metadatas: 메타데이터 리스트
            ids: 문서 ID 리스트 (없으면 자동 생성)
        """
        if not documents:
            return

        try:
            # 임베딩 생성 (Batch)
            embeddings = self.embedding_model.encode(documents, convert_to_numpy=True).tolist()
            
            # ID 생성 (없으면)
            if ids is None:
                import uuid
                ids = [str(uuid.uuid4()) for _ in documents]
            
            # ChromaDB에 추가
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"[VectorStore] Saved {len(documents)} documents to {self.collection_name}.")
            
        except Exception as e:
            print(f"[VectorStore][ERROR] add_documents failed: {e}")
            raise e

    def similarity_search(self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        유사도 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            filter: 메타데이터 필터링
            
        Returns:
            List[Dict]: {page_content: str, metadata: dict, distance: float}
        """
        try:
            # 쿼리 임베딩
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True).tolist()
            
            # 검색
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter, # 메타데이터 필터
                include=["documents", "metadatas", "distances"]
            )
            
            # 결과 변환
            # ChromaDB query returns list of lists (one per query)
            docs = []
            if results["documents"]:
                for i in range(len(results["documents"][0])):
                    docs.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0.0
                    })
                
            return docs
            
        except Exception as e:
            print(f"[VectorStore][ERROR] similarity_search failed: {e}")
            return []
