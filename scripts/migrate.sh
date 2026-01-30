#!/bin/bash
# ===========================================
# AI 상담 챗봇 - DB 마이그레이션 스크립트
# ===========================================
#
# 사용법:
#   ./scripts/migrate.sh
#
# 현재는 SQLite 사용, Phase 3에서 PostgreSQL 지원 추가 예정
# ===========================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo ""
echo "========================================"
echo "  DB 마이그레이션"
echo "========================================"
echo ""

# 가상환경 활성화
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "[OK] 가상환경 활성화됨"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "[OK] 가상환경 활성화됨"
fi

# .env 로드
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "[OK] 환경변수 로드됨"
fi

# DB 초기화 (SQLite)
echo "[MIGRATE] 데이터베이스 초기화 중..."

python -c "
from config.db_config import db_settings

# 디렉토리 생성
db_settings.ensure_directories()
print(f'[OK] 디렉토리 생성 완료')
print(f'     SQLite: {db_settings.SQLITE_DB_PATH}')
print(f'     ChromaDB: {db_settings.CHROMA_DB_DIR}')
"

echo ""
echo "[OK] 마이그레이션 완료!"
echo ""
