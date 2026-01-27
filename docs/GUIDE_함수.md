# Vector DB 함수 가이드

## Import
```python
from src.data import (
    # Save
    load_counseling_to_db,
    load_batch_to_db,
    
    # Load
    search_similar,
    get_all_documents,
    get_by_ids,
    get_document_count
)
```

---

## Save 함수

### `load_counseling_to_db(data)`
단일 상담 데이터 저장

```python
data = {
    'source_id': 'D012',
    'category': 'DEPRESSION',
    'metadata': {'age': 31, 'gender': '남', ...},
    'paragraphs': [{'speaker': '상담사', 'content': '...', 'index': 0}, ...],
    'labels': [{'depressive_mood': 0}, ...]
}
result = load_counseling_to_db(data)
# → {'counseling_id': 1, 'paragraphs_saved': 537}
```

### `load_batch_to_db(data_list)`
여러 상담 데이터 일괄 저장

```python
stats = load_batch_to_db([data1, data2, data3])
# → {'total': 3, 'success': 3, 'paragraphs': 1000, 'error': 0}
```

---

## Load 함수

### `search_similar(query, n_results, category, speaker)`
유사 상담 사례 검색

```python
# 기본 검색
results = search_similar("요즘 우울해요", n_results=5)

# 카테고리 필터
results = search_similar("불안해요", category="ANXIETY")

# 화자 필터
results = search_similar("힘들어요", speaker="내담자")

# 결과 형태
[
    {'content': '발화 내용', 'metadata': {...}, 'distance': 0.2},
    ...
]
```

### `get_all_documents(limit)`
전체 문서 조회

```python
docs = get_all_documents(limit=100)
# → [{'id': 'abc', 'content': '...', 'metadata': {...}}, ...]
```

### `get_by_ids(ids)`
ID로 문서 조회

```python
docs = get_by_ids(['abc123', 'def456'])
# → [{'id': 'abc123', 'content': '...', 'metadata': {...}}, ...]
```

### `get_document_count()`
문서 수 조회

```python
count = get_document_count()
# → 11543
```

---

## 카테고리 값
- `DEPRESSION` - 우울증
- `ANXIETY` - 불안장애
- `ADDICTION` - 중독
- `NORMAL` - 일반군

## 화자 값
- `상담사`
- `내담자`

---

# SQLite DB 함수

## Import
```python
from src.data import (
    get_all_counselings,
    get_counseling_by_id,
    get_counseling_by_source_id,
    get_counselings_by_category,
    get_paragraphs_by_counseling,
    get_db_statistics
)
```

---

## 상담 데이터 조회

### `get_all_counselings(limit)`
모든 상담 데이터 조회

```python
counselings = get_all_counselings(limit=100)
# → [{'id': 1, 'source_id': 'D012', 'category': 'DEPRESSION', ...}, ...]
```

### `get_counseling_by_id(counseling_id)`
ID로 상담 조회

```python
counseling = get_counseling_by_id(1)
```

### `get_counseling_by_source_id(source_id)`
source_id로 상담 조회

```python
counseling = get_counseling_by_source_id("D012")
```

### `get_counselings_by_category(category)`
카테고리별 조회

```python
depressions = get_counselings_by_category("DEPRESSION")
```

---

## 발화 조회

### `get_paragraphs_by_counseling(counseling_id)`
특정 상담의 모든 발화 조회

```python
paragraphs = get_paragraphs_by_counseling(1)
# → [{'speaker': '상담사', 'content': '...', 'index': 0}, ...]
```

### `get_paragraphs_by_speaker(speaker)`
화자별 발화 조회

```python
client_talks = get_paragraphs_by_speaker("내담자")
```

---

## 통계

### `get_db_statistics()`
DB 전체 통계

```python
stats = get_db_statistics()
# → {'counseling_data': 41, 'counseling_paragraphs': 11543, ...}
```
