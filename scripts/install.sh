#!/bin/bash
# ===========================================
# AI 상담 챗봇 - 설치 스크립트 (Linux/Mac)
# ===========================================
#
# 사용법:
#   ./scripts/install.sh
#
# ===========================================

set -e

# 스크립트 위치 기준으로 프로젝트 루트로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo ""
echo "========================================"
echo "  AI 상담 챗봇 - 설치 스크립트"
echo "========================================"
echo ""

# Python 버전 확인
echo "[CHECK] Python 버전 확인..."
python3 --version || {
    echo "[ERROR] Python 3이 설치되지 않았습니다."
    exit 1
}

# ------------------------------------
# 가상환경 생성
# ------------------------------------
if [ ! -d "venv" ]; then
    echo "[CREATE] 가상환경 생성 중..."
    python3 -m venv venv
    echo "[OK] 가상환경 생성 완료"
else
    echo "[OK] 가상환경이 이미 존재합니다"
fi

# ------------------------------------
# 가상환경 활성화
# ------------------------------------
echo "[ACTIVATE] 가상환경 활성화..."
source venv/bin/activate

# ------------------------------------
# pip 업그레이드
# ------------------------------------
echo "[UPGRADE] pip 업그레이드..."
pip install --upgrade pip

# ------------------------------------
# 의존성 설치
# ------------------------------------
echo "[INSTALL] 의존성 설치 중..."
pip install -r requirements.txt

# ------------------------------------
# .env 파일 설정
# ------------------------------------
if [ ! -f ".env" ]; then
    echo ""
    echo "[WARN] .env 파일이 없습니다."
    
    if [ -f ".env.example" ]; then
        read -p "   .env.example을 .env로 복사할까요? (y/N): " response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            cp .env.example .env
            echo "[OK] .env 파일이 생성되었습니다."
            echo ""
            echo "=========================================="
            echo "  중요: .env 파일을 수정하세요!"
            echo "=========================================="
            echo "  특히 다음 값들을 설정하세요:"
            echo "    - OPENAI_API_KEY"
            echo "    - SECRET_KEY (프로덕션용)"
            echo ""
        fi
    fi
else
    echo "[OK] .env 파일이 이미 존재합니다"
fi

# ------------------------------------
# 완료
# ------------------------------------
echo ""
echo "========================================"
echo "  설치 완료!"
echo "========================================"
echo ""
echo "  실행 방법:"
echo "    ./start.sh              # 개발 서버"
echo "    ./start.sh --check      # 환경 검증"
echo ""
