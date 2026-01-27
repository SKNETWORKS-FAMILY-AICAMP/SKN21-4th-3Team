# 데이터베이스 설계 문서

---

## 1. 아키텍처 개요

```mermaid
flowchart TB
    subgraph 사용자영역["🖥️ 사용자 영역"]
        USER[사용자]
    end
    
    subgraph 애플리케이션["⚙️ 애플리케이션 계층"]
        FLASK[Flask API]
        RAG[RAG Chain]
        LLM[LLM Client]
    end
    
    subgraph 데이터계층["💾 데이터 계층"]
        subgraph SQLite["SQLite (관계형 DB)"]
            USERS[(users)]
            SESSIONS[(chat_sessions)]
            MESSAGES[(chat_messages)]
            COUNSELING[(counseling_data)]
            PARAGRAPHS[(counseling_paragraphs)]
            REFERRALS[(expert_referrals)]
        end
        
        subgraph ChromaDB["ChromaDB (벡터 DB)"]
            VECTORS[(psych_counseling_vectors)]
        end
    end
    
    USER --> FLASK
    FLASK --> RAG
    RAG --> LLM
    RAG --> VECTORS
    FLASK --> SESSIONS
    FLASK --> MESSAGES
    PARAGRAPHS -.-> VECTORS
```

---

## 2. ERD (Entity Relationship Diagram)

```mermaid
erDiagram
    users {
        int id PK "자동 증가"
        string username "사용자명"
        string password_hash "암호화 비밀번호 (선택)"
        datetime created_at "생성일시"
        datetime last_login "마지막 로그인"
    }
    
    chat_sessions {
        int id PK "자동 증가"
        int user_id FK "users.id"
        datetime started_at "시작 시간"
        datetime ended_at "종료 시간 (NULL 가능)"
        string status "active/completed/referred"
        json screening_result "증상 선별 결과"
    }
    
    chat_messages {
        int id PK "자동 증가"
        int session_id FK "chat_sessions.id"
        string role "user/assistant/system"
        text content "메시지 내용"
        datetime created_at "생성 시간"
    }
    
    counseling_data {
        int id PK "자동 증가"
        string source_id UK "원본 ID (D012)"
        string category "DEPRESSION/ANXIETY/ADDICTION/NORMAL"
        int severity "0-3 심각도 (NULL 허용)"
        text summary "상담 요약"
        string source_file "원본 파일 경로"
        string data_format "labeled/unlabeled"
        bool has_detailed_label "상세 라벨 존재 여부"
        json raw_metadata "원본 메타데이터"
        datetime imported_at "임포트 시간"
    }
    
    counseling_paragraphs {
        int id PK "자동 증가"
        int counseling_id FK "counseling_data.id"
        int paragraph_index "단락 순서"
        string speaker "상담사/내담자"
        text content "발화 내용"
        json labels "심리학적 라벨 (NULL 허용)"
        string vector_id "ChromaDB 문서 ID"
    }
    
    expert_referrals {
        int id PK "자동 증가"
        int session_id FK "chat_sessions.id (1:1)"
        string severity_level "mild/moderate/severe/crisis"
        text recommended_action "권장 조치"
        datetime created_at "생성 시간"
    }
    
    users ||--o{ chat_sessions : "1:N 보유"
    chat_sessions ||--o{ chat_messages : "1:N 포함"
    chat_sessions ||--o| expert_referrals : "1:0..1 연결"
    counseling_data ||--o{ counseling_paragraphs : "1:N 포함"
```

---

## 3. 테이블 설명

### 3.1 사용자 관련

| 테이블 | 용도 | 비고 |
|--------|------|------|
| **users** | 사용자 정보 | 익명 사용 가능 |
| **chat_sessions** | 채팅 세션 | 상태 추적 (active/completed/referred) |
| **chat_messages** | 대화 기록 | role로 화자 구분 |
| **expert_referrals** | 전문가 연결 | 세션당 최대 1회 |

### 3.2 상담 데이터

| 테이블 | 용도 | 비고 |
|--------|------|------|
| **counseling_data** | 상담 세션 원본 | 메타데이터 포함 |
| **counseling_paragraphs** | 발화 단위 분할 | Vector DB와 연동 |

---

## 4. 데이터 흐름

```mermaid
sequenceDiagram
    participant U as 사용자
    participant API as Flask API
    participant DB as SQLite
    participant RAG as RAG Chain
    participant VDB as ChromaDB
    participant LLM as OpenAI
    
    U->>API: 메시지 전송
    API->>DB: 메시지 저장 (chat_messages)
    API->>RAG: 질의 전달
    RAG->>VDB: 유사 상담 검색
    VDB-->>RAG: 관련 단락 반환
    RAG->>LLM: 컨텍스트 + 질문
    LLM-->>RAG: 응답 생성
    RAG-->>API: 응답 반환
    API->>DB: 응답 저장
    API-->>U: 응답 표시
```

---

## 5. 주요 파일

| 파일 | 설명 |
|------|------|
| [db_config.py](file:///c:/SKN21-3rd-3Team/config/db_config.py) | DB 경로 설정 |
| [database_schema.py](file:///c:/SKN21-3rd-3Team/src/database/database_schema.py) | ORM 모델 정의 |
| [vector_store.py](file:///c:/SKN21-3rd-3Team/src/database/vector_store.py) | ChromaDB 래퍼 |
| [db_manager.py](file:///c:/SKN21-3rd-3Team/src/database/db_manager.py) | 통합 CRUD |

---

## 6. 사용 예시

```python
from src.database import DatabaseManager

# 초기화
db = DatabaseManager()

# 상담 데이터 저장
counseling = db.add_counseling_data(
    source_id="D012",
    category="DEPRESSION",
    severity=2
)

# 단락 저장 (자동으로 VectorDB에도 저장)
db.add_counseling_paragraph(
    counseling_id=counseling.id,
    paragraph_index=0,
    speaker="내담자",
    content="요즘 너무 우울해요..."
)

# 유사 상담 검색
results = db.search_similar_counseling("우울한 기분", n_results=5)
```
