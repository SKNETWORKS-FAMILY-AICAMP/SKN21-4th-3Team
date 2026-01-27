"""
FileName    : __init__.py
Auth        : 수빈
Date        : 2026-01-03
Description : data 패키지 초기화 - 주요 클래스 export
Issue/Note  : 
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
    get_all_counselings,
    get_counseling_by_id,
    get_counseling_by_source_id,
    get_counselings_by_category,
    get_counseling_count,
    get_paragraphs_by_counseling,
    get_paragraphs_by_speaker,
    get_paragraph_count,
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
    "get_all_counselings",
    "get_counseling_by_id",
    "get_counseling_by_source_id",
    "get_counselings_by_category",
    "get_counseling_count",
    "get_paragraphs_by_counseling",
    "get_paragraphs_by_speaker",
    "get_paragraph_count",
    "get_db_statistics"
]
