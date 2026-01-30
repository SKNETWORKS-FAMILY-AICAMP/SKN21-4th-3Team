#!/bin/bash
# ===========================================
# AI 상담 챗봇 - 실행 스크립트 (Linux/Mac)
# ===========================================
#
# 사용법:
#   ./start.sh              개발 서버 실행
#   ./start.sh --prod       프로덕션 서버 실행
#   ./start.sh --check      환경 검증만 수행
#
# ===========================================

set -e

# 스크립트 위치로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "========================================"
echo "  AI 상담 챗봇"
echo "========================================"
echo ""

# 가상환경 활성화 (있는 경우)
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "[OK] 가상환경 활성화됨 (venv)"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "[OK] 가상환경 활성화됨 (.venv)"
else
    echo "[WARN] 가상환경이 없습니다. 시스템 Python 사용"
fi

# .env 파일 확인
if [ ! -f ".env" ]; then
    echo ""
    echo "[WARN] .env 파일이 없습니다."
    echo "   .env.example을 복사하여 설정하세요:"
    echo ""
    echo "   cp .env.example .env"
    echo "   nano .env"
    echo ""
    
    # .env.example 복사 여부 확인
    if [ -f ".env.example" ]; then
        read -p "   .env.example을 .env로 복사할까요? (y/N): " response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            cp .env.example .env
            echo "   [OK] .env 파일이 생성되었습니다. 값을 수정하세요."
        fi
    fi
fi

# 서버 실행
echo ""
python run.py "$@"
