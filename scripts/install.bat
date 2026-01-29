@echo off
REM ===========================================
REM AI 상담 챗봇 - 설치 스크립트 (Windows)
REM ===========================================
REM
REM 사용법:
REM   scripts\install.bat
REM
REM ===========================================

REM 스크립트 위치로 이동
cd /d "%~dp0\.."

echo.
echo ========================================
echo   AI 상담 챗봇 - 설치 스크립트
echo ========================================
echo.

REM Python 버전 확인
echo [CHECK] Python 버전 확인...
python --version
if errorlevel 1 (
    echo [ERROR] Python이 설치되지 않았습니다.
    pause
    exit /b 1
)

REM ------------------------------------
REM 가상환경 생성
REM ------------------------------------
if not exist "venv" (
    echo [CREATE] 가상환경 생성 중...
    python -m venv venv
    echo [OK] 가상환경 생성 완료
) else (
    echo [OK] 가상환경이 이미 존재합니다
)

REM ------------------------------------
REM 가상환경 활성화
REM ------------------------------------
echo [ACTIVATE] 가상환경 활성화...
call venv\Scripts\activate.bat

REM ------------------------------------
REM pip 업그레이드
REM ------------------------------------
echo [UPGRADE] pip 업그레이드...
pip install --upgrade pip

REM ------------------------------------
REM 의존성 설치
REM ------------------------------------
echo [INSTALL] 의존성 설치 중...
pip install -r requirements.txt

REM ------------------------------------
REM .env 파일 설정
REM ------------------------------------
if not exist ".env" (
    echo.
    echo [WARN] .env 파일이 없습니다.
    
    if exist ".env.example" (
        set /p response="   .env.example을 .env로 복사할까요? (y/N): "
        if /i "%response%"=="y" (
            copy .env.example .env
            echo [OK] .env 파일이 생성되었습니다.
            echo.
            echo ==========================================
            echo   중요: .env 파일을 수정하세요!
            echo ==========================================
            echo   특히 다음 값들을 설정하세요:
            echo     - OPENAI_API_KEY
            echo     - SECRET_KEY ^(프로덕션용^)
            echo.
        )
    )
) else (
    echo [OK] .env 파일이 이미 존재합니다
)

REM ------------------------------------
REM 완료
REM ------------------------------------
echo.
echo ========================================
echo   설치 완료!
echo ========================================
echo.
echo   실행 방법:
echo     start.bat              # 개발 서버
echo     start.bat --check      # 환경 검증
echo.
pause
