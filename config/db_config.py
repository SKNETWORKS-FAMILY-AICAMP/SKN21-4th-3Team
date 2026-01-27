"""
FileName    : db_config.py
Auth        : 박수빈
Date        : 2026-01-03
Description : 데이터베이스 설정 파일 - SQLite 및 ChromaDB 경로, 환경 변수 관리
Issue/Note  : .env 파일에서 API 키 로드, 환경별 설정 분리
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

import os
from pathlib import Path
from dotenv import load_dotenv

# -------------------------------------------------------------
# Environment Setup
# -------------------------------------------------------------

# .env 파일 로드
load_dotenv()

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# -------------------------------------------------------------
# Database Configuration
# -------------------------------------------------------------

class DatabaseConfig:
    """
    데이터베이스 설정을 관리하는 클래스
    SQLite와 ChromaDB의 경로 및 설정을 중앙에서 관리
    """
    
    # SQLite 설정 - 심리 상담 데이터베이스
    SQLITE_DB_NAME = "mind_care.db"
    SQLITE_DB_PATH = DATA_DIR / SQLITE_DB_NAME
    
    # ChromaDB 설정 - 벡터 임베딩 저장소
    CHROMA_DB_DIR = DATA_DIR / "vector_store"
    CHROMA_COLLECTION_NAME = "psych_counseling_vectors"
    
    # 데이터 디렉토리
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    @classmethod
    def get_sqlite_url(cls) -> str:
        """
        SQLAlchemy용 SQLite 연결 URL 반환
        """
        return f"sqlite:///{cls.SQLITE_DB_PATH}"
    
    @classmethod
    def ensure_directories(cls) -> None:
        """
        필요한 디렉토리들이 존재하는지 확인하고, 없으면 생성
        """
        directories = [
            DATA_DIR,
            cls.CHROMA_DB_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Entry Point (테스트용)
# -------------------------------------------------------------

if __name__ == "__main__":
    # 디렉토리 생성 테스트
    DatabaseConfig.ensure_directories()
    print(f"SQLite DB Path: {DatabaseConfig.SQLITE_DB_PATH}")
    print(f"ChromaDB Dir: {DatabaseConfig.CHROMA_DB_DIR}")