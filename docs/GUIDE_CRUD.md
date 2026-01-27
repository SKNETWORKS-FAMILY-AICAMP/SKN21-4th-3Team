# SQLite & ChromaDB CRUD 가이드

## Import
```python
from src.database import DatabaseManager
```

---

# SQLite CRUD

## Create (생성)

### 상담 데이터 추가
```python
db = DatabaseManager()

counseling = db.add_counseling_data(
    source_id="D012",
    category="DEPRESSION",
    summary="상담 요약...",
    severity=2,
    source_file="경로/file.txt",
    has_detailed_label=True,
    raw_metadata={"age": 31, "gender": "남"}
)
print(f"생성된 ID: {counseling.id}")
```

### 발화 데이터 추가
```python
db.add_counseling_paragraph(
    counseling_id=1,
    paragraph_index=0,
    speaker="상담사",
    content="안녕하세요.",
    labels={"depressive_mood": 0},
    category="DEPRESSION",
    severity=2
)
```

---

## Read (조회)

```python
from src.data import (
    get_all_counselings,
    get_counseling_by_id,
    get_paragraphs_by_counseling,
    get_db_statistics
)

# 전체 조회
counselings = get_all_counselings(limit=10)

# ID로 조회
counseling = get_counseling_by_id(1)

# 상담별 발화 조회
paragraphs = get_paragraphs_by_counseling(1)

# 통계
stats = get_db_statistics()
```

---

## Update (수정)

```python
from sqlalchemy.orm import Session
from src.database.database_schema import CounselingData
from src.data.db_loader import get_db_session

session = get_db_session()
counseling = session.query(CounselingData).filter_by(id=1).first()
counseling.summary = "수정된 요약"
session.commit()
session.close()
```

---

## Delete (삭제)

```python
session = get_db_session()
# 발화 먼저 삭제 (FK 제약)
session.query(CounselingParagraph).filter_by(counseling_id=1).delete()
# 상담 데이터 삭제
session.query(CounselingData).filter_by(id=1).delete()
session.commit()
session.close()
```

---

# ChromaDB CRUD

## Create (생성)

```python
from src.database import VectorStore

vs = VectorStore()
ids = vs.add_documents(
    documents=["발화 내용 1", "발화 내용 2"],
    metadatas=[
        {"speaker": "상담사", "category": "DEPRESSION"},
        {"speaker": "내담자", "category": "DEPRESSION"}
    ]
)
```

---

## Read (조회)

```python
from src.data import search_similar, get_all_documents, get_document_count

# 유사 검색
results = search_similar("우울해요", n_results=5)

# 전체 조회
docs = get_all_documents(limit=100)

# 문서 수
count = get_document_count()
```

---

## Update (수정)

```python
# ChromaDB는 직접 수정 불가
# 삭제 후 재추가
vs = VectorStore()
vs.delete_documents(ids=["abc123"])
vs.add_documents(documents=["수정된 내용"], metadatas=[{...}])
```

---

## Delete (삭제)

```python
vs = VectorStore()

# 특정 문서 삭제
vs.delete_documents(ids=["abc123", "def456"])

# 전체 삭제
vs.clear_collection()
```
