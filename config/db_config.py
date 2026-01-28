"""
FileName    : db_config.py
Auth        : 박수빈, 우재현
Date        : 2026-01-28
Description : 데이터베이스 설정 파일 - SQLite 및 ChromaDB 경로, 환경 변수 관리 (Pydantic applied)
Issue/Note  : Pydantic Settings 적용
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import ClassVar

class DatabaseSettings(BaseSettings):
    """
    데이터베이스 설정을 관리하는 클래스 (Pydantic BaseSettings)
    """
    # 환경 변수 설정
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # 기본 경로 설정
    # Pydantic에서는 이런 계산된 필드를 field_validator나 model_post_init에서 처리할 수도 있지만,
    # 여기서는 간단하게 ClassVar나 property를 사용하거나,
    # PROJECT_ROOT를 Path(__file__).parent.parent 로 고정하고 나머지 필드 기본값으로 설정합니다.

    # 프로젝트 루트는 이 파일(config/db_config.py)의 상위 상위 디렉토리
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    # SQLite
    SQLITE_DB_NAME: str = "mind_care.db"
    
    # ChromaDB
    CHROMA_COLLECTION_NAME: str = "psych_counseling_vectors"

    @property
    def DATA_DIR(self) -> Path:
        return self.PROJECT_ROOT / "data"

    @property
    def SQLITE_DB_PATH(self) -> Path:
        return self.DATA_DIR / self.SQLITE_DB_NAME

    @property
    def CHROMA_DB_DIR(self) -> Path:
        return self.DATA_DIR / "vector_store"

    @property
    def RAW_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "raw"

    @property
    def PROCESSED_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "processed"

    def get_sqlite_url(self) -> str:
        """
        SQLAlchemy용 SQLite 연결 URL 반환
        """
        return f"sqlite:///{self.SQLITE_DB_PATH}"

    def ensure_directories(self) -> None:
        """
        필요한 디렉토리들이 존재하는지 확인하고, 없으면 생성
        """
        directories = [
            self.DATA_DIR,
            self.CHROMA_DB_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# 전역 설정 인스턴스 생성
db_settings = DatabaseSettings()


if __name__ == "__main__":
    # 디렉토리 생성 테스트
    db_settings.ensure_directories()
    print(f"SQLite DB Path: {db_settings.SQLITE_DB_PATH}")
    print(f"ChromaDB Dir: {db_settings.CHROMA_DB_DIR}")