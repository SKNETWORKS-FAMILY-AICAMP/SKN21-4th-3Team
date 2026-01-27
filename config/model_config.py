"""
FileName    : model_config.py
Auth        : 박수빈, 우재현
Date        : 2026-01-05
Description : API_KEY, 모델 설정 파일
Issue/Note  : db_config로부터 OpenAIConfig 분리, .env 파일에서 API 키 로드, 환경별 설정 분리
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# -------------------------------------------------------------
# OpenAI Configuration
# -------------------------------------------------------------

class OpenAIConfig:
    """
    OpenAI API 설정
    """
    ENV_PATH = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=ENV_PATH)
    
    API_KEY = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL = "text-embedding-3-small"
    OPENAI_CHAT_MODEL=os.getenv("OPENAI_CHAT_MODEL", "")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

    @classmethod
    def validate(cls) -> bool:
        """
        API 키와 모델이 설정되어 있는지 확인
        """
        if not cls.API_KEY:
            print("WARNING: OPENAI_API_KEY가 설정되지 않았습니다.")
            return False
        if not cls.EMBEDDING_MODEL:
            print("WARNING: EMBEDDING_MODEL가 설정되지 않았습니다.")
            return False
        if not cls.OPENAI_CHAT_MODEL:
            print("WARNING: OPENAI_CHAT_MODEL가 설정되지 않았습니다.")
            return False
        if not cls.TAVILY_API_KEY:
            print("WARNING: TAVILY_API_KEY가 설정되지 않았습니다.")
            return False
        return True

# -------------------------------------------------------------
# Model Create Functions
# -------------------------------------------------------------

def create_chat_model() -> ChatOpenAI:
    """
    OpenAIConfig.OPENAI_CHAT_MODEL에서 모델이름을 가져와 ChatOpenAI 모델을 실질적으로 생성
    """
    model_name = OpenAIConfig.OPENAI_CHAT_MODEL
    print(f"모델이름: {model_name}")

    try:
        model = ChatOpenAI(model=model_name)
    except Exception as e:
        raise RuntimeError(
            f"ChatOpenAI 초기화 실패 (model={model_name})"
        ) from e

    return model

# -------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------

if __name__ == "__main__":
    print(f"OpenAI API Key Set: {OpenAIConfig.validate()}")
    chat_model=create_chat_model()
    print(type(chat_model))