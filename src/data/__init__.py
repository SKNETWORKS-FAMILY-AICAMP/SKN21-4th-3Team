"""
FileName    : __init__.py
Auth        : 수빈
Date        : 2026-01-03
Description : data 패키지 초기화 - 주요 클래스 export
Issue/Note  : [2026-01-28] db_loader 함수 정리 (기존 테이블 삭제로 인한 함수 제거)
"""

from src.data.vector_loader import (
    # Vector Save
    load_counseling_to_db, 
    load_batch_to_db,
    # Vector Load
    search_similar,
    get_all_documents,
    get_by_ids,
    get_document_count
)

from src.data.db_loader import (
    # SQLite Load
    get_db_session,
    get_db_statistics
)

__all__ = [
    # Vector Save
    "load_counseling_to_db",
    "load_batch_to_db",
    # Vector Load
    "search_similar",
    "get_all_documents",
    "get_by_ids",
    "get_document_count",
    # SQLite Load
    "get_db_session",
    "get_db_statistics"
]

