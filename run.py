#!/usr/bin/env python3
"""
FileName    : run.py
Auth        : 우재현
Date        : 2026-01-29
Description : AI 상담 챗봇 - 통합 실행 스크립트

사용법:
    python run.py              # 개발 서버 실행
    python run.py --prod       # 프로덕션 서버 실행 (Uvicorn)
    python run.py --check      # 환경 검증만 수행
    
Issue/Note  : 이 파일 하나로 개발/프로덕션 환경 모두 실행 가능
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------------------------------------------
# Functions
# -------------------------------------------------------------

def print_banner():
    """배너 출력"""
    print()
    print("=" * 50)
    print("  AI 상담 챗봇")
    print("=" * 50)
    print()

def check_environment() -> bool:
    """
    환경 검증
    Returns:
        bool: 검증 성공 여부
    """
    from config.settings import app_settings
    from config.db_config import db_settings
    
    print("[CHECK] 환경 검증 중...")
    print("-" * 40)
    
    # 앱 설정 출력
    print(f"  환경: {app_settings.FLASK_ENV}")
    print(f"  서버: {app_settings.HOST}:{app_settings.PORT}")
    print(f"  디버그: {app_settings.DEBUG}")
    print(f"  OpenAI API: {'[OK] 설정됨' if app_settings.OPENAI_API_KEY else '[ERROR] 미설정'}")
    
    # DB 설정 출력
    if db_settings.DATABASE_URL:
        # DATABASE_URL이 설정된 경우 (PostgreSQL 등 외부 DB)
        # URL에서 비밀번호 마스킹 처리
        db_url = db_settings.DATABASE_URL
        if "@" in db_url:
            # postgresql://user:password@host:port/db 형식에서 password 마스킹
            prefix = db_url.split("://")[0]  # postgresql
            rest = db_url.split("://")[1]    # user:password@host:port/db
            if ":" in rest.split("@")[0]:
                user = rest.split(":")[0]
                after_pass = rest.split("@")[1]
                db_url = f"{prefix}://{user}:****@{after_pass}"
        print(f"  Database: {db_url}")
        print(f"  ChromaDB: {db_settings.CHROMA_DB_DIR}")
    else:
        # 로컬 SQLite 사용
        print(f"  SQLite DB: {db_settings.SQLITE_DB_PATH}")
        print(f"  ChromaDB: {db_settings.CHROMA_DB_DIR}")
    
    print("-" * 40)
    
    # 검증
    errors = app_settings.validate_config()
    
    if errors:
        print("\n[ERROR] 오류 발견:")
        for e in errors:
            print(f"  - {e}")
        return False
    
    # 디렉토리 생성
    db_settings.ensure_directories()
    print("[OK] 환경 검증 완료!\n")
    return True

def run_development():
    """개발 서버 실행 (Flask 내장 서버)"""
    from config.settings import app_settings
    from app.main import app
    
    print(f"[START] 개발 서버 시작: http://{app_settings.HOST}:{app_settings.PORT}")
    print("   (Ctrl+C 로 종료)\n")
    
    app.run(
        host=app_settings.HOST,
        port=app_settings.PORT,
        debug=True
    )

def run_production():
    """프로덕션 서버 실행 (Uvicorn ASGI)"""
    from config.settings import app_settings
    
    try:
        import uvicorn
    except ImportError:
        print("[ERROR] uvicorn이 설치되지 않았습니다.")
        print("   설치: pip install uvicorn[standard]")
        sys.exit(1)
    
    print(f"[START] 프로덕션 서버 시작: http://{app_settings.HOST}:{app_settings.PORT}")
    print("   Workers: 4")
    print("   (Ctrl+C 로 종료)\n")
    
    uvicorn.run(
        "app.main:app",
        host=app_settings.HOST,
        port=app_settings.PORT,
        workers=4,
        log_level="info"
    )

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI 상담 챗봇",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python run.py              개발 서버 실행
  python run.py --prod       프로덕션 서버 실행
  python run.py --check      환경 검증만 수행
        """
    )
    parser.add_argument(
        "--prod", 
        action="store_true", 
        help="프로덕션 모드로 실행 (Uvicorn)"
    )
    parser.add_argument(
        "--check", 
        action="store_true", 
        help="환경 검증만 수행"
    )
    
    args = parser.parse_args()
    
    # 배너 출력
    print_banner()
    
    # 환경 검증
    if not check_environment():
        print("\n[WARN] 환경 검증 실패. .env 파일을 확인하세요.")
        print("   예시: cp .env.example .env")
        sys.exit(1)
    
    # 검증만 수행
    if args.check:
        return
    
    # 서버 실행
    try:
        if args.prod:
            run_production()
        else:
            run_development()
    except KeyboardInterrupt:
        print("\n\n[INFO] 서버가 종료되었습니다.")
    except Exception as e:
        print(f"\n[ERROR] 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
