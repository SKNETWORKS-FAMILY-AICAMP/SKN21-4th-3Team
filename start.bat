@echo off
REM ===========================================
REM AI 상담 챗봇 - 실행 스크립트 (Windows)
REM ===========================================
REM
REM 사용법:
REM   start.bat              개발 서버 실행
REM   start.bat --prod       프로덕션 서버 실행
REM   start.bat --check      환경 검증만 수행
REM
REM ===========================================

REM 스크립트 위치로 이동
cd /d "%~dp0"

echo.
echo ========================================
echo   AI 상담 챗봇
echo ========================================
echo.

REM 가상환경 활성화 (있는 경우)
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [OK] 가상환경 활성화됨 (venv)
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [OK] 가상환경 활성화됨 (.venv)
) else (
    echo [WARN] 가상환경이 없습니다. 시스템 Python 사용
)

REM .env 파일 확인
if not exist ".env" (
    echo.
    echo [WARN] .env 파일이 없습니다.
    echo        .env.example을 복사하여 설정하세요:
    echo.
    echo        copy .env.example .env
    echo        notepad .env
    echo.
    
    if exist ".env.example" (
        set /p response="       .env.example을 .env로 복사할까요? (y/N): "
        if /i "%response%"=="y" (
            copy .env.example .env
            echo        [OK] .env 파일이 생성되었습니다. 값을 수정하세요.
        )
    )
)

REM 서버 실행
echo.
python run.py %*

REM 오류 발생 시 대기
if errorlevel 1 (
    echo.
    echo [ERROR] 서버 실행 중 오류가 발생했습니다.
    pause
)
