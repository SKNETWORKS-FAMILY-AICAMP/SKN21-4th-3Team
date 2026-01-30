"""
FileName    : verify_refactor.py
Auth        : 우재현
Date        : 2026-01-28
Description : 설정 파일 리팩토링(Pydantic Settings 적용) 검증 스크립트
Issue/Note  : db_config, model_config 설정 및 VectorStore/DatabaseManager 연동 확인
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가 (tests 폴더의 상위 디렉토리)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# -------------------------------------------------------------
# Test Functions
# -------------------------------------------------------------

def test_settings():
    """
    설정(Settings) 로드 테스트
    - db_settings: SQLite/ChromaDB 설정 확인
    - model_settings: API 키 및 모델 설정 확인
    """
    print("--- 설정(Settings) 로드 테스트 ---")
    try:
        from config.db_config import db_settings
        from config.model_config import model_settings
        
        print(f"SQLITE_DB_NAME: {db_settings.SQLITE_DB_NAME}")
        print(f"CHROMA_COLLECTION_NAME: {db_settings.CHROMA_COLLECTION_NAME}")
        assert db_settings.SQLITE_DB_NAME == "mind_care.db"
        
        print(f"OPENAI_CHAT_MODEL: {model_settings.OPENAI_CHAT_MODEL}")
        # API 키 로드 확인 (길이 체크)
        has_key = bool(model_settings.OPENAI_API_KEY)
        print(f"OPENAI_API_KEY 존재 여부: {has_key}")
        
        print("결과: 성공 (PASS)")
        return True
    except Exception as e:
        print(f"결과: 실패 (FAILED) - {e}")
        return False

def test_vector_store():
    """
    VectorStore 초기화 테스트
    - 객체 생성 및 컬렉션 연동 확인
    - 문서 수 조회
    """
    print("\n--- VectorStore 초기화 테스트 ---")
    try:
        from src.database.vector_store import VectorStore
        
        # 초기화
        vs = VectorStore()
        print(f"VectorStore 초기화 완료. 컬렉션명: {vs.collection.name}")
        
        # 문서 수 확인
        count = vs.get_document_count()
        print(f"현재 문서 수: {count}")
        
        print("결과: 성공 (PASS)")
        return True
    except Exception as e:
        print(f"결과: 실패 (FAILED) - {e}")
        return False

def test_db_manager():
    """
    DatabaseManager 테스트
    - 객체 생성 및 통계 조회
    """
    print("\n--- DatabaseManager 테스트 ---")
    try:
        from src.database.db_manager import DatabaseManager
        
        db = DatabaseManager()
        stats = db.get_statistics()
        print(f"DB 통계: {stats}")
        db.close()
        
        print("결과: 성공 (PASS)")
        return True
    except Exception as e:
        print(f"결과: 실패 (FAILED) - {e}")
        return False

# -------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------

if __name__ == "__main__":
    results = [
        test_settings(),
        test_vector_store(),
        test_db_manager()
    ]
    
    if all(results):
        print("\n=== 모든 시스템 점검 완료: 정상 ===")
        sys.exit(0)
    else:
        print("\n=== 시스템 점검 실패: 오류 발생 ===")
        sys.exit(1)

