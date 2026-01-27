"""
FileName    : db_loader.py
Auth        : 박수빈
Date        : 2026-01-05
Description : SQLite DB 공통 함수 (저장 + 조회)
Issue/Note  : 기존 counseling_data, counseling_paragraphs 테이블 삭제됨
              현재는 users, chat_sessions, chat_messages, expert_referrals만 사용
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from config.db_config import DatabaseConfig
from src.database.database_schema import (
    User,
    ChatSession,
    ChatMessage,
    ExpertReferral
)

# -------------------------------------------------------------
# DB Connection
# -------------------------------------------------------------

def get_db_session() -> Session:
    """DB 세션 생성"""
    engine = create_engine(DatabaseConfig.get_sqlite_url())
    return Session(engine)


# -------------------------------------------------------------
# 통계
# -------------------------------------------------------------

def get_db_statistics() -> Dict[str, int]:
    """DB 전체 통계"""
    session = get_db_session()
    try:
        return {
            'users': session.query(User).count(),
            'chat_sessions': session.query(ChatSession).count(),
            'chat_messages': session.query(ChatMessage).count(),
            'expert_referrals': session.query(ExpertReferral).count()
        }
    finally:
        session.close()


# -------------------------------------------------------------
# Entry Point (테스트용)
# -------------------------------------------------------------

if __name__ == "__main__":
    print("=== SQLite DB 함수 테스트 ===\n")
    
    stats = get_db_statistics()
    print("1. DB 통계:")
    for k, v in stats.items():
        print(f"   - {k}: {v}")
