# 임베딩 처리 방법론 (Embedding Methodology)

## 개요

전처리된 상담 데이터를 벡터 데이터베이스(ChromaDB)에 임베딩하는 과정을 설명합니다.

---

## 📊 임베딩 통계

| 항목        | 값                            |
| ----------- | ----------------------------- |
| 총 문서 수  | 129,267개                     |
| 임베딩 모델 | `jhgan/ko-sroberta-multitask` |
| 벡터 차원   | 768                           |
| 저장소      | ChromaDB (SQLite 백엔드)      |
| 저장 용량   | ~1.6GB                        |

---

## 🎯 임베딩 모델 선택

### 선택 모델: `jhgan/ko-sroberta-multitask`

**선택 이유:**

1. **한국어 특화**
   - 한국어 문장 유사도 태스크에 최적화
   - SRoBERTa 아키텍처 기반 (Sentence-BERT의 RoBERTa 버전)

2. **멀티태스크 학습**
   - STS(Semantic Textual Similarity), NLI(Natural Language Inference) 등 다양한 태스크로 학습
   - 범용적인 문장 표현 생성

3. **성능 검증**
   - KorSTS 벤치마크에서 상위권 성능
   - 심리 상담 도메인의 감정적 뉘앙스 포착에 적합

4. **실용성**
   - 모델 크기: ~500MB (GPU 메모리 효율적)
   - 추론 속도: 빠름 (실시간 RAG 가능)

### 대안 모델 비교

| 모델                                            | 장점              | 단점               | 선택 이유 |
| ----------------------------------------------- | ----------------- | ------------------ | --------- |
| `jhgan/ko-sroberta-multitask`                   | 한국어 특화, 빠름 | -                  | ✅ 채택   |
| `sentence-transformers/paraphrase-multilingual` | 다국어 지원       | 한국어 성능 낮음   | ❌        |
| `BM-K/KoSimCSE-roberta`                         | 최신 기법         | 멀티태스크 미지원  | ❌        |
| `OpenAI text-embedding-3`                       | 최고 성능         | API 비용, 지연시간 | ❌        |

---

## 🔄 임베딩 파이프라인

```
docs_for_vectordb.jsonl
    ↓ 문서 로드
Document 객체 생성 (content + metadata)
    ↓ SentenceTransformer
768차원 벡터 생성
    ↓ ChromaDB
영구 저장 (data/vector_store/)
```

---

## 🏗️ 아키텍처 설계

### ChromaDB 선택 이유

1. **로컬 임베딩 저장**
   - 외부 서비스 의존 없음
   - SQLite 백엔드로 단일 파일 관리

2. **GPU 가속 지원**
   - CUDA 12.8+ (RTX 5060 Blackwell 지원)
   - 배치 임베딩 시 10배 이상 속도 향상

3. **메타데이터 필터링**
   - `category`, `speaker` 등으로 검색 범위 제한 가능

4. **LangChain 통합**
   - `Chroma.from_documents()` 등 편리한 API

### 벡터 저장 구조

```
data/vector_store/
├── chroma.sqlite3           # 메인 DB (인덱스 + 메타데이터)
└── {collection_id}/
    ├── data_level0.bin      # HNSW 인덱스
    ├── header.bin
    ├── length.bin
    └── link_lists.bin
```

---

## 🎯 검색 전략

### 유사도 측정: 코사인 유사도

```python
# ChromaDB 기본 설정
distance_fn = "cosine"
```

**이유:** 문장 임베딩에서 코사인 유사도가 유클리드 거리보다 의미적 유사성을 더 잘 반영

### 검색 파라미터

| 파라미터               | 값   | 설명                        |
| ---------------------- | ---- | --------------------------- |
| `top_k`                | 5    | 상위 5개 문서 반환          |
| `similarity_threshold` | 0.18 | 최소 유사도 (이하는 필터링) |

### 메타데이터 필터 활용

```python
# 특정 카테고리만 검색
results = collection.query(
    query_texts=["우울해요"],
    where={"category": "DEPRESSION"},
    n_results=5
)
```

---

## 💡 최적화 포인트

### 1. GPU 가속 설정

```python
# vector_store.py
self.ef = SentenceTransformerEmbeddingFunction(
    model_name="jhgan/ko-sroberta-multitask",
    device="cuda"  # RTX 5060 지원 (PyTorch 2.10.0+cu128)
)
```

### 2. 배치 처리

대량 문서 임베딩 시 배치 크기 조절로 메모리 효율화:

```python
batch_size = 1000  # GPU 메모리에 따라 조정
```

### 3. 영구 저장

```python
# PersistentClient 사용
client = chromadb.PersistentClient(path="data/vector_store")
```

---

## 🛠️ 실행 방법

### 초기 임베딩 (전체 데이터)

```bash
python src/data/vector_loader.py
```

### 증분 추가

```python
from src.data.vector_loader import load_counseling_to_db

load_counseling_to_db(new_data)
```

---

## ❓ 문제 해결

### CUDA 호환성 오류

RTX 5060 (Blackwell, sm_120) 사용 시:

```bash
pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

### 메모리 부족

배치 크기 줄이기 또는 CPU 모드로 전환:

```python
device="cpu"  # GPU 메모리 부족 시
```

---

## 📚 참고 자료

- **소스 코드**: `src/data/vector_loader.py`, `src/database/vector_store.py`
- **모델**: [jhgan/ko-sroberta-multitask](https://huggingface.co/jhgan/ko-sroberta-multitask)
- **ChromaDB 문서**: [docs.trychroma.com](https://docs.trychroma.com)
