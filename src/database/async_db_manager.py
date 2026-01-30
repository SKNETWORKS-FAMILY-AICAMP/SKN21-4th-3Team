"""
FileName    : async_db_manager.py
Auth        : 우재현
Date        : 2026-01-29
Description : 비동기 데이터베이스 매니저
              - SQLAlchemy 2.0 async 지원
              - PostgreSQL (asyncpg) / SQLite (aiosqlite) 호환
Issue/Note  : db_manager.py의 비동기 버전
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

from typing import Optional, List
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.db_config import db_settings
from src.database.database_schema import Base, User, ChatSession, ChatMessage, ExpertReferral


# -------------------------------------------------------------
# Async Database Manager Class
# -------------------------------------------------------------

class AsyncDatabaseManager:
    """
    비동기 데이터베이스 매니저
    
    SQLAlchemy 2.0의 async 기능을 활용하여
    PostgreSQL (asyncpg) 및 SQLite (aiosqlite) 지원
    """
    
    def __init__(self, echo: bool = False):
        """
        AsyncDatabaseManager 초기화
        
        Args:
            echo: SQL 쿼리 로그 출력 여부
        """
        self.engine = create_async_engine(
            db_settings.get_async_database_url(),
            echo=echo,
            pool_pre_ping=True
        )
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def init_tables(self):
        """
        테이블 생성 (비동기)
        """
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def close(self):
        """
        엔진 종료
        """
        await self.engine.dispose()
    
    # ---------------------------------------------------------
    # User CRUD
    # ---------------------------------------------------------
    
    async def create_user(self, username: str, password_hash: Optional[str] = None) -> User:
        """
        사용자 생성
        """
        async with self.async_session() as session:
            user = User(username=username, password_hash=password_hash)
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user
    
    async def get_user(self, user_id: int) -> Optional[User]:
        """
        ID로 사용자 조회
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """
        사용자명으로 사용자 조회
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(User.username == username)
            )
            return result.scalar_one_or_none()
    
    # ---------------------------------------------------------
    # Chat Session CRUD
    # ---------------------------------------------------------
    
    async def create_chat_session(self, user_id: int) -> ChatSession:
        """
        새 채팅 세션 생성
        """
        async with self.async_session() as session:
            chat_session = ChatSession(user_id=user_id)
            session.add(chat_session)
            await session.commit()
            await session.refresh(chat_session)
            return chat_session
    
    async def get_chat_session(self, session_id: int) -> Optional[ChatSession]:
        """
        채팅 세션 조회
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(ChatSession).where(ChatSession.id == session_id)
            )
            return result.scalar_one_or_none()
    
    # ---------------------------------------------------------
    # Chat Message CRUD
    # ---------------------------------------------------------
    
    async def add_chat_message(self, session_id: int, role: str, content: str) -> ChatMessage:
        """
        채팅 메시지 추가
        """
        async with self.async_session() as session:
            message = ChatMessage(
                session_id=session_id,
                role=role,
                content=content
            )
            session.add(message)
            await session.commit()
            await session.refresh(message)
            return message
    
    async def get_chat_history(self, session_id: int) -> List[ChatMessage]:
        """
        세션의 채팅 히스토리 조회
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at)
            )
            return list(result.scalars().all())
    
    async def get_user_recent_sessions(self, user_id: int, limit: int = 5) -> List[dict]:
        """
        사용자의 최근 채팅 세션 목록 조회
        
        Args:
            user_id: 사용자 ID
            limit: 반환할 세션 수
            
        Returns:
            세션 목록 (id, title, date, started_at)
        """
        async with self.async_session() as session:
            # 세션 조회
            result = await session.execute(
                select(ChatSession)
                .where(ChatSession.user_id == user_id)
                .order_by(ChatSession.started_at.desc())
                .limit(limit)
            )
            sessions = result.scalars().all()
            
            recent_sessions = []
            for chat_session in sessions:
                # 첫 번째 user 메시지 조회
                msg_result = await session.execute(
                    select(ChatMessage)
                    .where(ChatMessage.session_id == chat_session.id)
                    .where(ChatMessage.role == 'user')
                    .order_by(ChatMessage.created_at)
                    .limit(1)
                )
                first_msg = msg_result.scalar_one_or_none()
                
                title = first_msg.content[:30] + "..." if first_msg and len(first_msg.content) > 30 else (first_msg.content if first_msg else "새 대화")
                
                recent_sessions.append({
                    'id': chat_session.id,
                    'title': title,
                    'date': chat_session.started_at.strftime('%Y-%m-%d') if chat_session.started_at else None,
                    'started_at': chat_session.started_at.isoformat() if chat_session.started_at else None
                })
            
            return recent_sessions
    
    # ---------------------------------------------------------
    # Expert Referral CRUD
    # ---------------------------------------------------------
    
    async def create_expert_referral(
        self, 
        session_id: int, 
        severity_level: str, 
        recommended_action: Optional[str] = None
    ) -> ExpertReferral:
        """
        전문가 연결 기록 생성 또는 업데이트
        """
        async with self.async_session() as session:
            # 기존 레코드 확인
            result = await session.execute(
                select(ExpertReferral).where(ExpertReferral.session_id == session_id)
            )
            existing = result.scalar_one_or_none()
            
            if existing:
                existing.severity_level = severity_level
                existing.recommended_action = recommended_action
                await session.commit()
                await session.refresh(existing)
                return existing
            else:
                referral = ExpertReferral(
                    session_id=session_id,
                    severity_level=severity_level,
                    recommended_action=recommended_action
                )
                session.add(referral)
                await session.commit()
                await session.refresh(referral)
                return referral


# -------------------------------------------------------------
# Entry Point (테스트용)
# -------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("AsyncDatabaseManager 테스트...")
        
        db = AsyncDatabaseManager(echo=True)
        await db.init_tables()
        print("테이블 초기화 완료")
        
        # 사용자 생성 테스트
        user = await db.create_user("async_test_user")
        print(f"사용자 생성: {user}")
        
        # 채팅 세션 생성
        chat_session = await db.create_chat_session(user.id)
        print(f"채팅 세션 생성: {chat_session}")
        
        # 메시지 추가
        msg = await db.add_chat_message(chat_session.id, "user", "비동기 테스트 메시지")
        print(f"메시지 추가: {msg}")
        
        await db.close()
        print("\n비동기 테스트 완료!")
    
    asyncio.run(test())
