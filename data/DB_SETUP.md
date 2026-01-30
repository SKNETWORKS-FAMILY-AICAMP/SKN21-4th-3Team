# 데이터베이스 설정 가이드 (Database Setup Guide)

이 프로젝트의 데이터베이스 파일들은 용량이 커서 Git에 포함되지 않습니다.
아래 가이드를 따라 데이터를 설정하세요.

---

## 📦 데이터 파일 구조

```
data/
├── mind_care.db              # 사용자/세션 DB (자동 생성됨)
├── raw/                      # 원본 상담 데이터 (2,650개 파일)
├── processed/
│   ├── docs_for_vectordb.jsonl   # 벡터DB용 문서 (367MB)
│   └── sessions_OK.jsonl         # 전처리된 세션 (569MB)
└── vector_store/
    ├── chroma.sqlite3            # ChromaDB (1.6GB)
    └── {collection_id}/          # 임베딩 데이터
```

**총 용량: 약 2.5GB**

---

## 🔗 데이터 다운로드

### 방법 1: Google Drive (권장)

1. [데이터 다운로드 링크](https://drive.google.com/YOUR_LINK_HERE)에서 `mind_care_data.zip` 다운로드
2. 프로젝트 루트에 압축 해제:
   ```powershell
   Expand-Archive -Path mind_care_data.zip -DestinationPath .
   ```

### 방법 2: 팀 공유 폴더

팀 내부 NAS 또는 공유 드라이브에서 `data/` 폴더 전체를 복사하세요.

---

## ⚙️ 설치 후 확인

```powershell
# 가상환경 활성화
.\.venv\Scripts\Activate.ps1

# 서버 실행
cd app
python main.py
```

정상 실행 시 아래와 같은 메시지가 출력됩니다:

```
[INFO] Found collection: psych_counseling_vectors, documents: 129267
[INFO] VectorDB loaded
[INFO] RAG 시스템 초기화 완료
```

---

## 🗜️ 데이터 압축 방법 (배포용)

배포용으로 데이터를 압축할 때는 아래 명령어를 사용하세요:

### Windows PowerShell

```powershell
# 프로젝트 루트에서 실행
Compress-Archive -Path "data\processed", "data\vector_store" -DestinationPath "mind_care_data.zip" -CompressionLevel Optimal
```

### 7-Zip (더 높은 압축률)

```powershell
7z a -t7z -mx=9 mind_care_data.7z data\processed data\vector_store
```

> ⚠️ `data/raw/`는 전처리 완료 후 불필요하므로 제외해도 됩니다.

---

## ❓ 문제 해결

### "No module named 'sentence_transformers'" 에러

```powershell
.\.venv\Scripts\pip.exe install sentence-transformers
```

### "CUDA error: no kernel image" 에러

RTX 5060 (Blackwell) 사용 시 CUDA 12.8+ PyTorch 필요:

```powershell
.\.venv\Scripts\pip.exe install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

---

## 📁 .gitignore 설정

아래 항목들이 `.gitignore`에 포함되어 있어야 합니다:

```gitignore
# Data (큰 파일은 Git에서 제외)
data/raw/
data/processed/*.jsonl
data/vector_store/
*.db
*.sqlite3
```
