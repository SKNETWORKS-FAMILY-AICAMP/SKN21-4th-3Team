"""
FileName    : intent_router.py
Auth        : 박수빈
Date        : 2026-01-28
Description : 사용자 쿼리 의도 분류 및 라우팅
              RAG 검색 전에 쿼리 의도를 분류하여 처리 경로를 분기
Issue/Note  : GREETING/CHITCHAT은 RAG 없이 직접 응답
              EMOTION/QUESTION은 RAG 파이프라인으로 전달
              CRISIS는 즉시 전문가 연결 안내
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.model_config import create_chat_model

# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------

class QueryIntent(Enum):
    """쿼리 의도 분류"""
    GREETING = "GREETING"       # 인사 (안녕, 반가워요)
    CHITCHAT = "CHITCHAT"       # 잡담 (날씨, 시간 등)
    EMOTION = "EMOTION"         # 감정 표현/고민 (힘들어, 우울해)
    QUESTION = "QUESTION"       # 정보 질문 (우울증이란?)
    CRISIS = "CRISIS"           # 위기 상황 (자해, 자살 언급)
    CLOSING = "CLOSING"         # 상담 종료 (그만할래, 종료, 끝내고 싶어, 수고했어)


# 직접 응답이 가능한 의도들 (RAG 불필요)
DIRECT_RESPONSE_INTENTS = {QueryIntent.GREETING, QueryIntent.CHITCHAT, QueryIntent.CLOSING}

# RAG 검색이 필요한 의도들
RAG_REQUIRED_INTENTS = {QueryIntent.EMOTION, QueryIntent.QUESTION}

# 즉시 위기 대응이 필요한 의도
CRISIS_INTENTS = {QueryIntent.CRISIS}


# 의도 분류 프롬프트
INTENT_CLASSIFICATION_PROMPT = """\
당신은 심리상담 챗봇의 의도 분류기입니다.
사용자의 발화를 분석하여 아래 5가지 중 하나로 분류하세요.

[의도 카테고리]
- GREETING: 인사 (안녕, 하이, 반가워, 좋은 아침, 뭐해)
- CHITCHAT: 일상 잡담, 상담과 무관한 대화, 단순 명사/고유명사 언급 (날씨, 시간, 음식, 오바마, 트럼프, 영화, 연예인)
- EMOTION: 감정 표현, 고민 토로, 심리적 어려움 (힘들어, 우울해, 불안해, 스트레스, 짜증나)
- QUESTION: 심리/상담 관련 정보 질문 (우울증이란?, 불안장애 증상, 상담 방법)
- CRISIS: 자해/자살 언급, 극단적 위기 상황 (죽고싶어, 자해, 끝내고 싶어)
- CLOSING: 상담 종료 요청 (그만할래, 이제 나갈게, 상담 종료, 수고했어, 고마워)

[규칙]
1. 반드시 위 6개 중 하나만 출력하세요.
2. 감정 표현이 있으면 EMOTION으로 분류하세요.
3. 위기 키워드가 있으면 무조건 CRISIS로 분류하세요.
4. **상담과 관련 없는 인물(오바마 등), 정치, 사회, 단순 사실 언급은 CHITCHAT으로 분류하세요.**
5. 종료 의사가 명확하면 CLOSING으로 분류하세요.
6. 정말 애매하거나 모르겠다면 CHITCHAT으로 분류하세요.

[예시]
- "안녕" → GREETING
- "오늘 날씨 어때?" → CHITCHAT
- "오바마" → CHITCHAT
- "요즘 너무 힘들어" → EMOTION
- "우울증 증상이 뭐야?" → QUESTION
- "더 이상 살고 싶지 않아" → CRISIS
- "짜증나" → EMOTION
- "아이유" → CHITCHAT
- "불안해서 잠이 안 와" → EMOTION
"""

USER_PROMPT = """\
사용자 발화: {query}

위 발화의 의도를 분류하세요. (GREETING, CHITCHAT, EMOTION, QUESTION, CRISIS 중 하나)
의도:"""


# 직접 응답 템플릿
DIRECT_RESPONSES = {
    QueryIntent.GREETING: [
        "안녕하세요! 오늘 기분은 어떠세요? 😊",
        "반가워요! 무엇이든 편하게 이야기해 주세요.",
        "안녕하세요! 오늘 하루는 어떠셨나요?"
    ],
    QueryIntent.CHITCHAT: [
        "저는 심리 상담을 도와드리는 AI예요. 일상적인 대화보다는 회원님의 고민을 듣고 싶어요. 요즘 마음이 힘드신 일이 있으신가요?"
    ],
    QueryIntent.CRISIS: [
        "지금 많이 힘드시군요. 당신의 이야기를 듣고 있어요.\n\n"
        "혼자 감당하기 어려우시다면, 전문 상담사와 이야기해 보시는 것을 권해드려요.\n"
        "📞 자살예방상담전화: 1393 (24시간)\n"
        "📞 정신건강위기상담전화: 1577-0199\n\n"
        "전화하기 어려우시면, 저와 조금 더 이야기 나눠볼까요?"
    ]
}


# -------------------------------------------------------------
# Intent Classification Chain
# -------------------------------------------------------------

def create_intent_chain(model=None):
    """
    LLM 기반 의도 분류 체인 생성
    """
    if model is None:
        model = create_chat_model()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", INTENT_CLASSIFICATION_PROMPT),
        ("user", USER_PROMPT)
    ])
    
    chain = prompt | model | StrOutputParser()
    return chain


def classify_intent(query: str, model=None) -> QueryIntent:
    """
    사용자 쿼리의 의도를 분류합니다.
    
    Args:
        query: 사용자 입력 텍스트
        model: LLM 모델 (없으면 자동 생성)
    
    Returns:
        QueryIntent 열거형 값
    """
    # 1. 빠른 규칙 기반 분류 (키워드 매칭)
    quick_intent = _quick_classify(query)
    if quick_intent is not None:
        print(f"[Intent] Quick classify: {quick_intent.value}")
        return quick_intent
    
    # 2. LLM 기반 분류
    try:
        chain = create_intent_chain(model)
        result = chain.invoke({"query": query})
        result = result.strip().upper()
        
        # 결과 파싱
        for intent in QueryIntent:
            if intent.value in result:
                print(f"[Intent] LLM classify: {intent.value}")
                return intent
        
        # 기본값: EMOTION (상담 맥락에서 안전한 선택)
        print(f"[Intent] Fallback to EMOTION (raw: {result})")
        return QueryIntent.EMOTION
        
    except Exception as e:
        print(f"[Intent] Classification error: {e}")
        return QueryIntent.EMOTION


def _quick_classify(query: str) -> Optional[QueryIntent]:
    """
    키워드 기반 빠른 의도 분류 (LLM 호출 없이)
    """
    q = query.strip().lower()
    
    # CRISIS 체크 (최우선)
    crisis_keywords = ["죽고", "자살", "자해", "끝내고", "죽을", "안 살고", "살기 싫"]
    for kw in crisis_keywords:
        if kw in q:
            return QueryIntent.CRISIS
    
    # GREETING 체크
    greeting_patterns = ["안녕", "반가", "하이", "헬로", "좋은 아침", "좋은 저녁"]
    if len(q) <= 10:  # 짧은 인사
        for pat in greeting_patterns:
            if pat in q:
                return QueryIntent.GREETING
    
    # EMOTION 체크 (감정 키워드)
    emotion_keywords = ["힘들", "우울", "불안", "슬프", "외롭", "짜증", "화나", "스트레스", 
                        "무기력", "지쳤", "피곤", "걱정", "두렵", "무섭"]
    for kw in emotion_keywords:
        if kw in q:
            return QueryIntent.EMOTION
    
    # LLM 분류 필요
    return None


def get_direct_response(intent: QueryIntent) -> Optional[str]:
    """
    RAG 없이 직접 응답을 생성합니다.
    
    Args:
        intent: 분류된 의도
    
    Returns:
        직접 응답 문자열 (RAG 필요 시 None)
    """
    import random
    
    if intent in DIRECT_RESPONSES:
        responses = DIRECT_RESPONSES[intent]
        return random.choice(responses)
    
    return None


def should_use_rag(intent: QueryIntent) -> bool:
    """
    해당 의도에 대해 RAG 검색이 필요한지 판단합니다.
    """
    return intent in RAG_REQUIRED_INTENTS


def route_query(query: str, model=None, history: List[Dict[str, str]] = None) -> Tuple[QueryIntent, Optional[str], bool]:
    """
    쿼리를 분류하고 라우팅 정보를 반환합니다.
    
    Args:
        query: 사용자 입력
        model: LLM 모델
        history: 대화 히스토리 (향후 맥락 기반 분류에 활용)
    
    Returns:
        Tuple of (intent, direct_response, needs_rag)
        - intent: 분류된 의도
        - direct_response: 직접 응답 (RAG 불필요 시) or None
        - needs_rag: RAG 검색 필요 여부
    """
    print(f"\n[Router] Query: {query}")
    if history:
        print(f"[Router] History context: {len(history)} messages")
    
    intent = classify_intent(query, model)
    
    if intent in CRISIS_INTENTS:
        # 위기 상황: 직접 응답 + RAG도 수행 (추가 맥락 위해)
        direct = get_direct_response(intent)
        print(f"[Router] Intent: {intent.value} | Needs RAG: False (CRISIS)")
        return intent, direct, False
    
    if intent in DIRECT_RESPONSE_INTENTS:
        # 인사/잡담: 직접 응답만
        direct = get_direct_response(intent)
        print(f"[Router] Intent: {intent.value} | Needs RAG: False (DIRECT)")
        return intent, direct, False

    # EMOTION/QUESTION: RAG 필요
    print(f"[Router] Intent: {intent.value} | Needs RAG: True")
    return intent, None, True


# -------------------------------------------------------------
# Entry Point (Test)
# -------------------------------------------------------------

if __name__ == "__main__":
    print("=== Intent Router Test ===\n")
    
    test_queries = [
        "안녕",
        "오늘 날씨 어때?",
        "요즘 너무 힘들어",
        "우울증 증상이 뭐야?",
        "더 이상 살고 싶지 않아",
        "반가워요",
        "짜증나",
        "불안해서 잠이 안 와",
    ]
    
    for q in test_queries:
        intent, direct_resp, needs_rag = route_query(q)
        print(f"Query: '{q}'")
        print(f"  Intent: {intent.value}")
        print(f"  Needs RAG: {needs_rag}")
        if direct_resp:
            print(f"  Direct Response: {direct_resp[:50]}...")
        print()
