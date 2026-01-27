"""
FileName    : database_schema.py
Auth        : 박수빈
Date        : 2026-01-03
Description : SQLAlchemy ORM 모델 정의 - 사용자, 채팅 세션, 상담 데이터 등
Issue/Note  : SQLite 기반, JSON 필드는 SQLAlchemy의 JSON 타입 사용
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    create_engine, 
    Column, 
    Integer, 
    String, 
    Text, 
    DateTime, 
    Boolean, 
    ForeignKey,
    JSON
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.engine import Engine

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.db_config import DatabaseConfig

# -------------------------------------------------------------
# SQLAlchemy Base
# -------------------------------------------------------------

Base = declarative_base()

# -------------------------------------------------------------
# User Model
# -------------------------------------------------------------

class User(Base):
    """
    사용자 테이블
    - 익명 사용자도 지원 (username이 자동 생성될 수 있음)
    - password_hash는 선택적 (익명 사용 시 NULL)
    - 회원가입 시 수집하는 개인정보 포함
    """
    __tablename__ = "users"
    
    # ---------------------------------------------------------
    # 기본 인증 정보
    # ---------------------------------------------------------
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False, unique=True)  # 아이디
    password_hash = Column(String(255), nullable=True)          # Bcrypt 해시 비밀번호
    
    # ---------------------------------------------------------
    # 개인 정보 (회원가입 시 입력)
    # ---------------------------------------------------------
    name = Column(String(50), nullable=True)           # 이름
    gender = Column(String(10), nullable=True)         # 성별 (male/female)
    birthdate = Column(String(10), nullable=True)      # 생년월일 (YYYY-MM-DD)
    phone = Column(String(20), nullable=True)          # 전화번호
    address = Column(String(255), nullable=True)       # 기본 주소 (RAG 상담 데이터 활용 가능)
    address_detail = Column(String(255), nullable=True) # 상세 주소
    
    # ---------------------------------------------------------
    # 타임스탬프
    # ---------------------------------------------------------
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    chat_sessions = relationship("ChatSession", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"


# -------------------------------------------------------------
# Chat Session Model
# -------------------------------------------------------------

class ChatSession(Base):
    """
    채팅 세션 테이블
    - 사용자와 챗봇 간의 대화 세션을 관리
    - screening_result: 증상 선별 결과 (JSON)
    - status: active(진행중), completed(완료), referred(전문가 연결됨)
    """
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    status = Column(String(20), default="active")  # active, completed, referred
    screening_result = Column(JSON, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session")
    expert_referral = relationship("ExpertReferral", back_populates="session", uselist=False)
    
    def __repr__(self):
        return f"<ChatSession(id={self.id}, user_id={self.user_id}, status='{self.status}')>"


# -------------------------------------------------------------
# Chat Message Model
# -------------------------------------------------------------

class ChatMessage(Base):
    """
    채팅 메시지 테이블
    - role: user(사용자), assistant(챗봇), system(시스템 메시지)
    """
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(10), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")
    
    def __repr__(self):
        return f"<ChatMessage(id={self.id}, role='{self.role}')>"





# -------------------------------------------------------------
# Expert Referral Model (전문가 연결)
# -------------------------------------------------------------

class ExpertReferral(Base):
    """
    전문가 연결 테이블
    - 증상 선별 결과에 따라 전문가 연결이 필요할 때 기록
    - severity_level: mild, moderate, severe, crisis
    """
    __tablename__ = "expert_referrals"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False, unique=True)
    severity_level = Column(String(20), nullable=False)  # mild, moderate, severe, crisis
    recommended_action = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("ChatSession", back_populates="expert_referral")
    
    def __repr__(self):
        return f"<ExpertReferral(id={self.id}, severity='{self.severity_level}')>"


# -------------------------------------------------------------
# Database Initialization
# -------------------------------------------------------------

def init_database(echo: bool = False) -> Engine:
    """
    데이터베이스 초기화 - 테이블 생성
    
    Args:
        echo: SQL 로그 출력 여부
    
    Returns:
        SQLAlchemy Engine 객체
    """
    # 디렉토리 생성
    DatabaseConfig.ensure_directories()
    
    # 엔진 생성
    engine = create_engine(DatabaseConfig.get_sqlite_url(), echo=echo)
    
    # 테이블 생성
    Base.metadata.create_all(engine)
    
    return engine


def get_session(engine: Engine):
    """
    SQLAlchemy 세션 생성
    """
    Session = sessionmaker(bind=engine)
    return Session()


# -------------------------------------------------------------
# Entry Point (테스트용)
# -------------------------------------------------------------

if __name__ == "__main__":
    print("데이터베이스 초기화 중...")
    engine = init_database(echo=True)
    print(f"데이터베이스 생성 완료: {DatabaseConfig.SQLITE_DB_PATH}")
    
    # 테스트 세션
    session = get_session(engine)
    print("세션 생성 완료")
    session.close()
