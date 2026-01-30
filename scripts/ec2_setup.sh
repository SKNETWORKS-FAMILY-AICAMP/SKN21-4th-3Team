#!/bin/bash
# ===========================================
# AI 상담 챗봇 - EC2 초기 설정
# ===========================================
#
# 사용법:
#   chmod +x ec2_setup.sh && ./ec2_setup.sh
#
# ===========================================
set -e

echo "========================================"
echo "  EC2 서버 초기 설정 시작"
echo "========================================"

# -------------------------------------------------------------
# 1. 시스템 패키지 업데이트
# -------------------------------------------------------------
echo ""
echo "[1/6] 시스템 패키지 업데이트..."
sudo yum update -y

# -------------------------------------------------------------
# 2. Python 3.11 설치
# -------------------------------------------------------------
echo ""
echo "[2/6] Python 3.11 설치..."
sudo yum install -y python3.11 python3.11-pip python3.11-devel

# Python 버전 확인
python3.11 --version

# -------------------------------------------------------------
# 3. 필수 패키지 설치
# -------------------------------------------------------------
echo ""
echo "[3/6] 필수 패키지 설치..."
sudo yum install -y git gcc libffi-devel

# -------------------------------------------------------------
# 4. 프로젝트 클론
# -------------------------------------------------------------
echo ""
echo "[4/6] 프로젝트 클론..."
cd /home/ec2-user

if [ -d "SKN21-4th-3Team" ]; then
    echo "기존 프로젝트 디렉토리가 존재합니다. 업데이트합니다..."
    cd SKN21-4th-3Team
    git pull origin main
else
    echo "프로젝트 클론 중..."
    git clone https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN21-4th-3Team.git
    cd SKN21-4th-3Team
fi

# -------------------------------------------------------------
# 5. 가상환경 및 의존성 설치
# -------------------------------------------------------------
echo ""
echo "[5/6] 가상환경 생성 및 의존성 설치..."
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# -------------------------------------------------------------
# 6. 환경변수 설정 안내
# -------------------------------------------------------------
echo ""
echo "[6/6] 환경변수 설정..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "[주의] .env 파일을 수정해주세요:"
    echo "  nano .env"
    echo ""
    echo "  필수 설정:"
    echo "    - OPENAI_API_KEY=sk-your-api-key"
    echo "    - SECRET_KEY=your-production-secret"
    echo "    - FLASK_ENV=production"
fi

# -------------------------------------------------------------
# 7. systemd 서비스 설치
# -------------------------------------------------------------
echo ""
echo "[7/7] systemd 서비스 설치..."
if [ -f "scripts/mindcare.service" ]; then
    sudo cp scripts/mindcare.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable mindcare
    echo "systemd 서비스 등록 완료!"
fi

echo ""
echo "========================================"
echo "  EC2 초기 설정 완료!"
echo "========================================"
echo ""
echo "다음 단계:"
echo "  1. .env 파일 수정: nano .env"
echo "  2. 서비스 시작: sudo systemctl start mindcare"
echo "  3. 상태 확인: sudo systemctl status mindcare"
echo "  4. 로그 확인: sudo journalctl -u mindcare -f"
echo ""
