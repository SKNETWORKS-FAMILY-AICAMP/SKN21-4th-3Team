"""
FileName    : settings.py
Auth        : 우재현
Date        : 2026-01-29
Description : 애플리케이션 전체 설정 관리 (Pydantic Settings)
              - 서버, 보안, API 키 등 앱 레벨 설정
              - db_config.py와 분리하여 역할 명확화
Issue/Note  : db_config.py는 DB 경로/설정 전담, 이 파일은 앱 설정 전담
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# -------------------------------------------------------------
# Settings Class
# -------------------------------------------------------------

class AppSettings(BaseSettings):
    """
    애플리케이션 전체 설정을 관리하는 클래스
    """
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # 프로젝트 루트
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    
    # -------------------------------------------
    # 환경 설정
    # -------------------------------------------
    FLASK_ENV: str = "development"
    
    @property
    def DEBUG(self) -> bool:
        return self.FLASK_ENV == "development"
    
    @property
    def IS_PRODUCTION(self) -> bool:
        return self.FLASK_ENV == "production"
    
    # -------------------------------------------
    # 서버 설정
    # -------------------------------------------
    HOST: str = "127.0.0.1"
    PORT: int = 5000
    
    # -------------------------------------------
    # 보안 설정
    # -------------------------------------------
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    
    # -------------------------------------------
    # API 키
    # -------------------------------------------
    OPENAI_API_KEY: Optional[str] = None
    
    # -------------------------------------------
    # 검증 메서드
    # -------------------------------------------
    def validate_config(self) -> list:
        """
        필수 설정 검증 - 오류 목록 반환
        """
        errors = []
        
        if not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        if self.IS_PRODUCTION:
            if self.SECRET_KEY == "dev-secret-key-change-in-production":
                errors.append("프로덕션 환경에서는 SECRET_KEY를 변경해야 합니다.")
        
        return errors
    
    def print_config(self) -> None:
        """
        현재 설정 출력 (디버깅용)
        """
        print("=" * 50)
        print("[INFO] 현재 설정")
        print("=" * 50)
        print(f"  환경: {self.FLASK_ENV}")
        print(f"  디버그: {self.DEBUG}")
        print(f"  호스트: {self.HOST}:{self.PORT}")
        print(f"  OpenAI API: {'[OK] 설정됨' if self.OPENAI_API_KEY else '[ERROR] 미설정'}")
        print("=" * 50)


# 전역 설정 인스턴스
app_settings = AppSettings()


# -------------------------------------------------------------
# Entry Point (테스트용)
# -------------------------------------------------------------

if __name__ == "__main__":
    app_settings.print_config()
    
    errors = app_settings.validate_config()
    if errors:
        print("\n[ERROR] 설정 오류:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\n[OK] 설정 검증 완료!")
