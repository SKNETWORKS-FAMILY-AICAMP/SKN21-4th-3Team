"""
FileName    : db_manager.py
Auth        : 박수빈
Date        : 2026-01-03
Description : 데이터베이스 관리자 클래스 - 초기화, 마이그레이션, CRUD 유틸리티
Issue/Note  : SQLite와 ChromaDB를 통합 관리
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.db_config import DatabaseConfig
from src.database.database_schema import (
    init_database, get_session, 
    User, ChatSession, ChatMessage, ExpertReferral
)
from src.database.vector_store import VectorStore

# -------------------------------------------------------------
# Database Manager Class
# -------------------------------------------------------------

class DatabaseManager:
    """
    통합 데이터베이스 관리자
    
    SQLite (관계형 DB)와 ChromaDB (벡터 DB)를 함께 관리
    """
    
    def __init__(self, echo: bool = False):
        """
        DatabaseManager 초기화
        
        Args:
            echo: SQL 쿼리 로그 출력 여부
        """
        # SQLite 초기화
        self.engine = init_database(echo=echo)
        self._session: Optional[Session] = None
        
        # ChromaDB 초기화
        self.vector_store = VectorStore()
    
    # -------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------
    
    @property
    def session(self) -> Session:
        """
        현재 세션 반환 (없으면 생성)
        """
        if self._session is None:
            self._session = get_session(self.engine)
        return self._session
    
    def close(self) -> None:
        """
        세션 종료
        """
        if self._session:
            self._session.close()
            self._session = None
    
    def commit(self) -> None:
        """
        변경사항 커밋
        """
        self.session.commit()
    
    def rollback(self) -> None:
        """
        변경사항 롤백
        """
        self.session.rollback()
    
    # -------------------------------------------------------------
    # User CRUD
    # -------------------------------------------------------------
    
    def create_user(self, username: str, password_hash: Optional[str] = None) -> User:
        """
        사용자 생성
        """
        user = User(username=username, password_hash=password_hash)
        self.session.add(user)
        self.commit()
        return user
    
    def get_user(self, user_id: int) -> Optional[User]:
        """
        ID로 사용자 조회
        """
        return self.session.query(User).filter(User.id == user_id).first()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        사용자명으로 사용자 조회
        """
        return self.session.query(User).filter(User.username == username).first()
    
    # -------------------------------------------------------------
    # Chat Session CRUD
    # -------------------------------------------------------------
    
    def create_chat_session(self, user_id: int) -> ChatSession:
        """
        새 채팅 세션 생성
        """
        session = ChatSession(user_id=user_id)
        self.session.add(session)
        self.commit()
        return session
    
    def get_chat_session(self, session_id: int) -> Optional[ChatSession]:
        """
        채팅 세션 조회
        """
        return self.session.query(ChatSession).filter(ChatSession.id == session_id).first()
    
    def add_chat_message(self, session_id: int, role: str, content: str) -> ChatMessage:
        """
        채팅 메시지 추가
        """
        message = ChatMessage(session_id=session_id, role=role, content=content)
        self.session.add(message)
        self.commit()
        return message
    
    def get_chat_history(self, session_id: int) -> List[ChatMessage]:
        """
        세션의 채팅 히스토리 조회
        """
        return self.session.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.created_at).all()
    
    def get_user_recent_sessions(self, user_id: int, limit: int = 5) -> List[Dict]:
        """
        사용자의 최근 채팅 세션 목록 조회
        세션의 첫 번째 user 메시지를 제목으로 사용
        
        Args:
            user_id: 사용자 ID
            limit: 반환할 세션 수 (기본 5개)
            
        Returns:
            세션 목록 (id, title, date, started_at)
        """
        from datetime import datetime, timedelta
        
        # 사용자의 최근 세션 조회 (최신순)
        sessions = self.session.query(ChatSession).filter(
            ChatSession.user_id == user_id
        ).order_by(ChatSession.started_at.desc()).limit(limit).all()
        
        result = []
        today = datetime.utcnow().date()
        yesterday = today - timedelta(days=1)
        
        for chat_session in sessions:
            # 첫 번째 user 메시지를 제목으로 사용
            first_message = self.session.query(ChatMessage).filter(
                ChatMessage.session_id == chat_session.id,
                ChatMessage.role == "user"
            ).order_by(ChatMessage.created_at).first()
            
            if first_message:
                # 제목: 첫 메시지를 20자로 자르고 "..."
                title = first_message.content[:20]
                if len(first_message.content) > 20:
                    title += "..."
            else:
                title = "새 대화"
            
            # 날짜 포맷팅
            session_date = chat_session.started_at.date()
            if session_date == today:
                date_str = "오늘"
            elif session_date == yesterday:
                date_str = "어제"
            else:
                date_str = chat_session.started_at.strftime("%m월 %d일")
            
            result.append({
                "id": chat_session.id,
                "title": title,
                "date": date_str,
                "started_at": chat_session.started_at.isoformat()
            })
        
        return result
    
    def delete_chat_session(self, session_id: int) -> bool:
        """
        채팅 세션 삭제 (관련 메시지도 함께 삭제)
        
        Args:
            session_id: 삭제할 세션 ID
            
        Returns:
            삭제 성공 여부
        """
        # 먼저 해당 세션의 메시지들 삭제
        self.session.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).delete()
        
        # 세션 삭제
        self.session.query(ChatSession).filter(
            ChatSession.id == session_id
        ).delete()
        
        self.commit()
        return True
    

    
    # -------------------------------------------------------------
    # Expert Referral
    # -------------------------------------------------------------
    
    def create_expert_referral(
        self,
        session_id: int,
        severity_level: str,
        recommended_action: Optional[str] = None
    ) -> ExpertReferral:
        """
        전문가 연결 기록 생성 또는 업데이트
        - 동일 session_id에 대해 기존 레코드가 있으면 업데이트
        - 없으면 새로 생성
        """
        # 기존 레코드 확인
        existing = self.session.query(ExpertReferral).filter(
            ExpertReferral.session_id == session_id
        ).first()
        
        if existing:
            # 기존 레코드 업데이트
            existing.severity_level = severity_level
            if recommended_action:
                existing.recommended_action = recommended_action
            referral = existing
        else:
            # 새 레코드 생성
            referral = ExpertReferral(
                session_id=session_id,
                severity_level=severity_level,
                recommended_action=recommended_action
            )
            self.session.add(referral)
        
        # 세션 상태 업데이트
        chat_session = self.get_chat_session(session_id)
        if chat_session:
            chat_session.status = "referred"
        
        self.commit()
        return referral
    
    # -------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------
    
    def get_statistics(self) -> Dict[str, int]:
        """
        데이터베이스 통계 정보
        """
        return {
            "users": self.session.query(User).count(),
            "chat_sessions": self.session.query(ChatSession).count(),
            "chat_messages": self.session.query(ChatMessage).count(),
            "expert_referrals": self.session.query(ExpertReferral).count(),
            "vector_documents": self.vector_store.get_document_count()
        }


# -------------------------------------------------------------
# Entry Point (테스트용)
# -------------------------------------------------------------

if __name__ == "__main__":
    print("DatabaseManager 테스트...")
    
    db = DatabaseManager(echo=False)
    
    # 사용자 생성 테스트
    user = db.create_user("test_user_001")
    print(f"사용자 생성: {user}")
    
    # 채팅 세션 생성 테스트
    session = db.create_chat_session(user.id)
    print(f"세션 생성: {session}")
    
    # 메시지 추가 테스트
    db.add_chat_message(session.id, "user", "안녕하세요, 요즘 우울한 기분이 들어요.")
    db.add_chat_message(session.id, "assistant", "안녕하세요. 우울한 기분이 드신다니 힘드셨겠네요.")
    
    # 통계 출력
    stats = db.get_statistics()
    print(f"\n데이터베이스 통계:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    db.close()
    print("\n테스트 완료!")
