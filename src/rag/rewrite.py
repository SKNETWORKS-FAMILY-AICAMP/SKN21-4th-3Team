"""
FileName    : rewrite.py
Auth        : 손현우, 우재현
Date        : 2026-01-05 ~ 2026-01-06
Description : 대화 히스토리를 반영한 검색용 쿼리를 생성하는 함수
Issue/Note  : 그거, 이전에, 아까 등의 대화 내용을 찾을 수 있도록 Prompt 작성해야한다.
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

from typing import List, Dict, Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.model_config import create_chat_model

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------

SYSTEM_PROMPT = """\
너는 RAG 챗봇의 Query Rewriter다.
목표: 사용자의 마지막 발화를, 대화 히스토리를 반영한 "검색용 단일 문장 쿼리"로 재작성한다.

[규칙]
1) 출력은 오직 한 줄의 한국어 쿼리만.
2) 5글자 이하의 짧은 표현은 그대로 출력 (예: 안녕, 힘들어, 슬퍼).
3) 감정 키워드(힘들어, 우울해, 불안해, 슬퍼, 외로워, 스트레스)는 반드시 보존.
4) 단순한 감정 표현이나 불만에 대해서는 '대처법', '해결책', '방법' 등의 솔루션 관련 단어를 절대 추가하지 않는다. (중요)
5) 지시어(그거/아까/저번/그럼 등)는 히스토리로 구체화.
6) 사실/정보를 새로 만들어내지 않는다.

[예시]
입력: "안녕" → 출력: "안녕"
입력: "힘들어" → 출력: "힘들어"
입력: "그냥 여러가지 일이 겹쳐서 스트레스가 쌓여" → 출력: "여러 일이 겹쳐서 스트레스가 쌓 이다" (X: "스트레스 대처법", "스트레스 해소 방법")
입력: "우울해" → 출력: "우울해"
입력: "요즘 너무 힘들어요" → 출력: "요즘 힘들다"
입력: "아까 그거" (히스토리: 우울증 관련) → 출력: "우울증 관련 정보"
입력: "그럼 이거 우울증이랑 관련 있어?" (히스토리: 불안) → 출력: "불안과 우울증의 관련성"
"""

DEFAULT_MAX_TURNS = 6

# -------------------------------------------------------------
# rewrite helper functions
# -------------------------------------------------------------

def format_history(history: List[Dict[str, str]]) -> str:
    """
    대화 이력을 프롬프트용 문자열로 변환
    """
    if not history:
        return "없음"
    
    formatted = []
    # 최근 N개 턴만 사용 (토큰 관리)
    recent_history = history[-DEFAULT_MAX_TURNS:]
    
    for msg in recent_history:
        role = "사용자" if msg.get("role") == "user" else "상담사"
        content = msg.get("content", "")
        formatted.append(f"{role}: {content}")
        
    return "\n".join(formatted)

def rule_based_fallback(query: str) -> str:
    """
    LLM 실패 시 기본 정리
    """
    # 간단한 특수문자 제거 등
    return query.strip()

# -------------------------------------------------------------
# LCEL Chain Factory
# -------------------------------------------------------------

def create_rewrite_chain(model):
    """
    LCEL 방식의 Rewrite Chain 생성
    Chain: Prompt | Model | StrOutputParser
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", """\
[대화 히스토리]
{history}

[사용자 마지막 발화]
{query}

위 규칙에 따라 "검색용 단일 문장 쿼리"로 재작성해라.
""")
    ])
    
    chain = prompt | model | StrOutputParser()
    return chain

# -------------------------------------------------------------
# Main Functions (Legacy Wrapper)
# -------------------------------------------------------------

def rewrite_query(
    history: List[Dict[str, str]],
    last_user: str,
    model=None,
) -> str:
    """
    대화 히스토리를 반영하여 검색용 쿼리로 재작성합니다.
    (LCEL create_rewrite_chain을 내부적으로 사용하는 래퍼 함수)
    """
    # 1) model 준비
    if model is None:
        try:
            model = create_chat_model()
        except Exception:
            model = None

    # 2) 모델 없으면 fallback
    if model is None:
        return rule_based_fallback(last_user)

    # 3) LLM rewrite
    if not history:
        return last_user

    hist_text = format_history(history)
    
    try:
        chain = create_rewrite_chain(model)
        text = chain.invoke({"history": hist_text, "query": last_user})
        
        # 안전장치: 여러 줄이면 첫 줄만 사용
        text = text.splitlines()[0].strip()
        # 따옴표 제거 등 최소 정리
        text = text.strip('"\'')
        return text
        
    except Exception as e:
        print(f"[Warning] Rewrite failed: {e}")
        return rule_based_fallback(last_user)


# -------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------

if __name__ == "__main__":
    # Test only
    print("=== Rewrite Chain Test ===")
    history_test = [
        {"role": "user", "content": "요즘 불안해서 잠이 안 와요."},
        {"role": "assistant", "content": "불안이 심할 때는 호흡과 수면 루틴을 점검해볼 수 있어요."},
    ]
    query_test = "그럼 이거 우울증이랑 관련 있어?"
    
    print(f"[Prev History] ... {history_test[-1]}")
    print(f"[Current Query] {query_test}")
    
    # 모델 & 체인 생성 테스트
    try:
        model = create_chat_model()
        chain = create_rewrite_chain(model)
        
        hist_text = format_history(history_test)
        result = chain.invoke({"history": hist_text, "query": query_test})
        result = result.splitlines()[0].strip().strip('"\'')
        
        print(f"\n[Rewritten]: {result}")
        
    except Exception as e:
        print(f"[Error] {e}")