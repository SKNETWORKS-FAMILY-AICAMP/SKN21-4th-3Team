# CHANGELOG

프로젝트 변경 이력을 기록합니다.

---

## [2026-01-28] 데이터 파이프라인 개선 (평가서 피드백 반영)

### 🔄 Modified

#### `src/data/preprocess_data.py`

- **내담자 질문 중심 임베딩으로 변경**
  - 기존: 윈도우 내 전체 대화를 하나의 text로 임베딩
  - 변경: 내담자 발화만 `text`로 임베딩, 맥락은 `metadata`로 저장
- **메타데이터 구조 강화**
  - `context_text`: 앞뒤 대화 맥락
  - `counselor_response`: 상담사 응답

### ✨ Added

#### `src/data/reset_and_rebuild.py` [NEW]

- 통합 리셋 + 재처리 스크립트
- 기존 데이터 삭제 → 전처리 → 임베딩 자동화

#### `tests/evaluate_rag.py` [NEW]

- Recall@k 평가 시스템
- 정량적 RAG 성능 측정

#### `tests/test_queries.json` [NEW]

- 평가용 테스트 쿼리 세트

---

## 변경 이유

SKN 21기 평가서 피드백 (88/100점):

> "임베딩 단계에서는 내담자의 질문 중심으로 벡터화를 수행하고,
> 해당 질문을 포함한 일정 범위의 대화 맥락을 메타데이터로 저장하는 방식을
> 적용하였다면 검색 정확도 측면에서 성능이 개선될 수도 있지 않을까 생각된다."

---

## ⚠️ Known Issue

- **Python 3.13 + ChromaDB 호환성 문제**
  - ChromaDB C extension import 오류 발생
  - 해결: Python 3.11 환경에서 실행 권장
