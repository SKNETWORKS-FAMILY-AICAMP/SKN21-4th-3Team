"""
FileName    : model_config.py
Auth        : 박수빈, 우재현
Date        : 2026-01-28
Description : API_KEY, 모델 설정 파일 (Pydantic applied)
Issue/Note  : Pydantic Settings 적용
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_openai import ChatOpenAI
from typing import Optional

class ModelSettings(BaseSettings):
    """
    OpenAI 및 모델 설정을 관리하는 클래스 (Pydantic BaseSettings)
    """
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    OPENAI_API_KEY: str
    TAVILY_API_KEY: str
    
    # 기본값 설정
    OPENAI_CHAT_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    @property
    def EMBEDDING_MODEL(self) -> str:
        return self.OPENAI_EMBEDDING_MODEL

# 전역 설정 인스턴스
model_settings = ModelSettings()

# ChromaDB 등 외부 라이브러리가 환경변수에서 API 키를 찾을 수 있도록 설정
os.environ["OPENAI_API_KEY"] = model_settings.OPENAI_API_KEY
os.environ["TAVILY_API_KEY"] = model_settings.TAVILY_API_KEY


def create_chat_model() -> ChatOpenAI:
    """
    model_settings.OPENAI_CHAT_MODEL에서 모델이름을 가져와 ChatOpenAI 모델을 실질적으로 생성
    """
    model_name = model_settings.OPENAI_CHAT_MODEL
    print(f"모델이름: {model_name}")

    try:
        model = ChatOpenAI(model=model_name, api_key=model_settings.OPENAI_API_KEY, streaming=True)
    except Exception as e:
        raise RuntimeError(
            f"ChatOpenAI 초기화 실패 (model={model_name})"
        ) from e

    return model

if __name__ == "__main__":
    print(f"OpenAI API Key Set: {bool(model_settings.OPENAI_API_KEY)}")
    try:
        chat_model = create_chat_model()
        print(type(chat_model))
    except Exception as e:
        print(f"Error creating chat model: {e}")