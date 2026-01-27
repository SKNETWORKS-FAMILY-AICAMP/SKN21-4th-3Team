# 데이터 전처리 및 청킹 가이드

## 목표
원본 txt/json 파일을 파싱하여 DB에 저장 가능한 형태로 변환

---

## 전체 Flow

```
[입력] txt + json 파일
        ↓
[STEP 1] txt 파싱 (청킹) → 발화 단위 분리
[STEP 2] json 파싱 → 메타데이터 + 라벨 추출
        ↓
[STEP 3] 데이터 통합
        ↓
[출력] DB 저장용 데이터
```

---

## STEP 1: txt 파싱 (청킹)

### 입력
```
상담사 : 안녕하세요. 오늘 기분은 어떠세요?
내담자 : 요즘 너무 힘들어요.
상담사 : 그렇군요. 언제부터 그런 느낌이 드셨나요?
```

### 파싱 패턴
```python
pattern = r'^(상담사|내담자)\s*:\s*(.+)$'
```

### 출력
```python
paragraphs = [
    {"speaker": "상담사", "content": "안녕하세요. 오늘 기분은 어떠세요?", "index": 0},
    {"speaker": "내담자", "content": "요즘 너무 힘들어요.", "index": 1},
    {"speaker": "상담사", "content": "그렇군요. 언제부터...", "index": 2}
]
```

---

## STEP 2: json 파싱

### 입력
```json
{
    "id": "D012",
    "age": 31,
    "gender": "남",
    "depression": 2,
    "anxiety": 0,
    "class": "DEPRESSION",
    "summary": "주요 증상: ...",
    "paragraph": [
        {"paragraph_speaker": "상담사", "depressive_mood": 0, ...},
        {"paragraph_speaker": "내담자", "depressive_mood": 1, ...}
    ]
}
```

### 출력
```python
metadata = {
    "id": "D012",
    "age": 31,
    "gender": "남",
    "depression": 2,
    "anxiety": 0,
    "class": "DEPRESSION",
    "summary": "주요 증상: ..."
}

labels = [
    {"depressive_mood": 0, "suicidal": 0, ...},  # index 0
    {"depressive_mood": 1, "suicidal": 0, ...}   # index 1
]
```

---

## STEP 3: 데이터 통합 (DB 저장 형태)

### counseling_data (상담 세션)
```python
{
    "source_id": "D012",           # metadata['id']
    "category": "DEPRESSION",       # metadata['class']
    "severity": 2,                  # depression+anxiety+addiction
    "summary": "주요 증상: ...",    # metadata['summary']
    "raw_metadata": {...}           # 전체 metadata
}
```

### counseling_paragraphs (발화 단위)
```python
[
    {
        "paragraph_index": 0,
        "speaker": "상담사",
        "content": "안녕하세요...",
        "labels": {"depressive_mood": 0, ...}
    },
    {
        "paragraph_index": 1,
        "speaker": "내담자", 
        "content": "요즘 너무 힘들어요.",
        "labels": {"depressive_mood": 1, ...}
    }
]
```

---

## STEP 4: DB 입력

전처리된 데이터는 **2개의 DB**에 분리 저장됩니다.

### DB 구분

| DB | 용도 | 저장 데이터 |
|----|------|------------|
| **SQLite** (관계형) | 구조화된 데이터 관리 | 메타데이터, 사용자 정보, 채팅 기록 |
| **ChromaDB** (벡터) | 유사도 검색 (RAG) | 발화 내용의 임베딩 벡터 |

### SQLite 저장 대상

```
counseling_data       ← 상담 세션 메타데이터
counseling_paragraphs ← 발화 단위 텍스트 + 라벨
users, chat_sessions, chat_messages ← 사용자/채팅 데이터
```

### ChromaDB 저장 대상

```
psych_counseling_vectors ← 발화 content의 임베딩 벡터
```

> **핵심**: `counseling_paragraphs.content`가 임베딩되어 ChromaDB에 저장됨

### 저장 코드 예시

```python
from src.database.db_manager import DatabaseManager

db = DatabaseManager()

# 1. SQLite에 상담 데이터 저장
counseling = db.add_counseling_data(
    source_id="D012",
    category="DEPRESSION",
    severity=2,
    summary="주요 증상: ..."
)

# 2. 단락 저장 → SQLite + ChromaDB 동시 저장
db.add_counseling_paragraph(
    counseling_id=counseling.id,
    paragraph_index=0,
    speaker="내담자",
    content="요즘 너무 우울해요...",  # → ChromaDB에 임베딩
    labels={"depressive_mood": 1}
)
```

### 데이터 흐름 요약

```
전처리 결과
    │
    ├─→ counseling_data      → SQLite
    │
    └─→ counseling_paragraphs
            ├─→ 텍스트/라벨   → SQLite
            └─→ content 임베딩 → ChromaDB
```

---

## 파일 경로

| 구분 | 경로 패턴 |
|------|----------|
| 원천 txt | `01.원천데이터/TS_{카테고리}_{회기}/resource_*.txt` |
| 라벨 json | `02.라벨링데이터/TL_{카테고리}_{회기}/label_*.json` |

---

## 카테고리 매핑

| 폴더명 | 코드 |
|--------|------|
| 001. 우울증 | DEPRESSION |
| 002. 불안장애 | ANXIETY |
| 003. 중독 | ADDICTION |
| 004. 일반군 | NORMAL |

---

## 참고

- 데이터 구조 상세: [DATA_ANALYSIS.md](./DATA_ANALYSIS.md)
- DB 스키마: [DATABASE_DESIGN.md](./DATABASE_DESIGN.md)
