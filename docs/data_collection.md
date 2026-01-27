# 데이터 수집 및 임베딩 가이드

## 1. 수집 데이터 개요

| 항목 | 내용 |
|------|------|
| 데이터 출처 | AI Hub 심리 상담 데이터셋 |
| 수집 방법 | 공개 데이터셋 다운로드 |
| 데이터 형식 | txt (원천), json (라벨) |
| 데이터 크기 | 약 1,000+ 상담 세션 |

---

## 2. 데이터 구조

### 2.1 원천 데이터 (txt)
```
상담사 : 안녕하세요. 오늘 기분은 어떠세요?
내담자 : 요즘 너무 힘들어요.
상담사 : 그렇군요. 언제부터 그런 느낌이 드셨나요?
```

### 2.2 라벨 데이터 (json)
```json
{
    "id": "D012",
    "age": 31,
    "gender": "남",
    "class": "DEPRESSION",
    "depression": 2,
    "paragraph": [...]
}
```

---

## 3. 전처리 과정

1. txt 파싱 → 발화 단위 분리
2. json 파싱 → 메타데이터 + 라벨 추출
3. 데이터 통합 → DB 저장 형태로 변환

> 상세 내용: [GUIDE_전처리.md](./GUIDE_전처리.md)

---

## 4. 임베딩 (Embedding)

### 4.1 임베딩이란?
텍스트를 고차원 벡터로 변환하여 **의미적 유사도 검색**을 가능하게 하는 과정

```
"요즘 우울해요" → [0.12, -0.34, 0.56, ...] (1536차원 벡터)
```

### 4.2 사용 모델

| 항목 | 설정 |
|------|------|
| 모델 | OpenAI `text-embedding-3-small` |
| 차원 | 1536 |
| 벡터 DB | ChromaDB |

### 4.3 임베딩 대상

**`counseling_paragraphs.content`** (발화 내용)만 임베딩됨

```python
# 예시: 임베딩되는 데이터
documents = [
    "요즘 너무 우울해요.",           # 내담자 발화
    "언제부터 그런 느낌이 드셨나요?",  # 상담사 발화
    "일주일 정도 된 것 같아요."       # 내담자 발화
]
```

### 4.4 메타데이터

각 임베딩에 함께 저장되는 정보:

```python
metadata = {
    "counseling_id": 1,           # 상담 세션 ID
    "paragraph_index": 0,         # 발화 순서
    "speaker": "내담자",           # 화자
    "category": "DEPRESSION",     # 카테고리
    "severity": 2                 # 심각도 (0-3)
}
```

### 4.5 임베딩 코드 예시

#### 1) VectorStore 초기화
```python
from src.database.vector_store import VectorStore

# 기본 경로 사용 (data/vector_store)
vector_store = VectorStore()

# 커스텀 경로 지정
vector_store = VectorStore(persist_directory="./my_vectors")
```

#### 2) 단일 문서 임베딩 추가
```python
# 하나의 발화 저장
vector_store.add_documents(
    documents=["요즘 너무 우울해요."],
    metadatas=[{
        "counseling_id": 1,
        "paragraph_index": 0,
        "speaker": "내담자",
        "category": "DEPRESSION",
        "severity": 2
    }]
)
```

#### 3) 배치 임베딩 추가
```python
# 여러 발화를 한 번에 저장
documents = [
    "요즘 너무 우울해요.",
    "언제부터 그런 느낌이 드셨나요?",
    "일주일 정도 된 것 같아요.",
    "특별한 계기가 있으셨나요?"
]

metadatas = [
    {"counseling_id": 1, "paragraph_index": 0, "speaker": "내담자", "category": "DEPRESSION", "severity": 2},
    {"counseling_id": 1, "paragraph_index": 1, "speaker": "상담사", "category": "DEPRESSION", "severity": 2},
    {"counseling_id": 1, "paragraph_index": 2, "speaker": "내담자", "category": "DEPRESSION", "severity": 2},
    {"counseling_id": 1, "paragraph_index": 3, "speaker": "상담사", "category": "DEPRESSION", "severity": 2},
]

# 배치 저장 (한 번의 API 호출)
ids = vector_store.add_documents(documents=documents, metadatas=metadatas)
print(f"저장된 문서 ID: {ids}")
```

#### 4) 유사도 검색
```python
# 기본 검색
results = vector_store.search(
    query="우울한 기분이 들어요",
    n_results=5
)

# 결과 구조
# results = {
#     "ids": [["id1", "id2", ...]],
#     "documents": [["문서1", "문서2", ...]],
#     "metadatas": [[{...}, {...}, ...]],
#     "distances": [[0.1, 0.2, ...]]  # 거리 (낮을수록 유사)
# }
```

#### 5) 검색 결과 파싱
```python
results = vector_store.search("불안해요", n_results=3)

# 결과 출력
for i, (doc, meta, dist) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
)):
    print(f"[{i+1}] 유사도: {1 - dist:.2f}")
    print(f"    화자: {meta['speaker']}")
    print(f"    내용: {doc}")
    print(f"    카테고리: {meta['category']}")
    print()
```

#### 6) 필터링 검색
```python
# 카테고리 필터
results = vector_store.search_by_category(
    query="힘들어요",
    category="DEPRESSION",
    n_results=5
)

# 화자 필터 (상담사 응답만 검색)
results = vector_store.search_by_speaker(
    query="공감 표현",
    speaker="상담사",
    n_results=5
)

# 고위험 케이스 필터
results = vector_store.search_high_severity(
    query="자해 생각",
    min_severity=2,
    n_results=5
)
```

#### 7) 문서 관리
```python
# 저장된 문서 수 확인
count = vector_store.get_document_count()
print(f"총 {count}개 문서")

# 특정 문서 삭제
vector_store.delete_documents(ids=["doc_001", "doc_002"])

# 전체 삭제 (주의!)
vector_store.clear_collection()
```

#### 8) DatabaseManager 통합 사용 (권장)
```python
from src.database.db_manager import DatabaseManager

# DatabaseManager가 SQLite + ChromaDB 동시 관리
db = DatabaseManager()

# 상담 데이터 저장 (SQLite)
counseling = db.add_counseling_data(
    source_id="D012",
    category="DEPRESSION",
    severity=2,
    summary="우울증 상담 사례"
)

# 단락 저장 → SQLite + ChromaDB 자동 동기화
db.add_counseling_paragraph(
    counseling_id=counseling.id,
    paragraph_index=0,
    speaker="내담자",
    content="요즘 너무 우울해요.",  # 자동으로 임베딩 저장
    labels={"depressive_mood": 1}
)

# 유사 상담 검색 (ChromaDB)
results = db.search_similar_counseling("우울한 기분", n_results=5)
```

---

## 5. 전체 데이터 흐름

```
[원본 데이터]
txt + json 파일
      ↓
[전처리]
발화 파싱 + 메타데이터 추출
      ↓
[DB 저장]
      ├─→ SQLite: counseling_data, counseling_paragraphs
      │           (구조화된 데이터)
      │
      └─→ ChromaDB: content 임베딩 + 메타데이터
                    (벡터 검색용)
      ↓
[RAG 활용]
사용자 질문 → 유사 상담 검색 → LLM 응답 생성
```

---

## 6. 검색 API

### 6.1 기본 검색
```python
results = vector_store.search("우울해요", n_results=5)
```

### 6.2 카테고리별 검색
```python
results = vector_store.search_by_category(
    query="불안해요",
    category="ANXIETY",
    n_results=5
)
```

### 6.3 화자별 검색
```python
results = vector_store.search_by_speaker(
    query="상담 기법",
    speaker="상담사",
    n_results=5
)
```

### 6.4 고위험 케이스 검색
```python
results = vector_store.search_high_severity(
    query="자해 생각",
    min_severity=2,
    n_results=5
)
```

---

## 7. 관련 파일

| 파일 | 설명 |
|------|------|
| [vector_store.py](file:///c:/SKN21-3rd-3Team/src/database/vector_store.py) | ChromaDB 래퍼 (임베딩/검색) |
| [db_config.py](file:///c:/SKN21-3rd-3Team/config/db_config.py) | DB 경로 및 모델 설정 |
| [db_manager.py](file:///c:/SKN21-3rd-3Team/src/database/db_manager.py) | 통합 CRUD (SQLite + ChromaDB) |
| [GUIDE_전처리.md](./GUIDE_전처리.md) | 전처리 상세 가이드 |
