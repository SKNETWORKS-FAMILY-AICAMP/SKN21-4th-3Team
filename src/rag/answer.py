"""
FileName    : answer.py
Auth        : 우재현
Date        : 2026-01-05 ~ 2026-01-07
Description : RAG 시스템의 답변 생성, 대화내용 저장 모듈
Issue/Note  : docs/DATABASE_DESIGN.md를 참고, src/database/db_manager.py 내용을 반영하여서 함수를 제작.
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

import os
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from typing import List, Dict, Optional, Any

from src.database.vector_store import VectorStore
from src.database import db_manager
from config.model_config import create_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------

SYSTEM_PROMPT = """\
[역할]
당신은 '마음챙김' 심리 상담 AI입니다.
RAG(검색 증강 생성)를 통해 이전 상담 내역을 참고하여 답변합니다.

[RAG 검색 결과 처리 규칙 - 매우 중요]
1. 제공된 Context에 "[전문가 상담 가이드]"가 포함되어 있다면:
   - **가장 높은 우선순위**로 해당 가이드의 **구체적인 상담 기법(예: 점수 매기기, 호흡법, 특정 질문법 등)**을 답변에 적용하세요.
   - **금지 사항:** 검색된 기법이 있는데도 불구하고 "산책하세요", "음악을 들으세요" 같은 뻔하고 일반적인 조언을 하지 마세요.
   - **[문맥 자연스럽게 다듬기 - 필수]**: 검색된 텍스트의 어투가 대화의 흐름과 맞지 않거나 어색하다면(예: "그걸 계속 체크하는 게요"), **반드시 자연스럽고 정중한 상담 톤으로 수정**하여 표현하세요.
   - 검색된 사례에서 상담사가 0~10점 척도를 사용했다면, 당신도 정확히 그 기법을 사용하여 질문하세요.
   - 단순히 내용을 요약하는 것이 아니라, 당신이 그 전문 상담사가 된 것처럼 그 기법을 실연(Role-play)하세요.

2. 제공된 Context가 "관련된 상담 내역이 없습니다."라고 되어 있다면:
   - 검색 결과를 무시하고, 당신의 전문적인 심리 상담 지식을 바탕으로 공감하고 조언하세요.
   - "검색 결과가 없다"는 말을 사용자에게 절대 하지 마세요.

[최우선 규칙]
1. 응답은 3~5문장 정도로, 충분한 공감과 제안을 담으세요. 따옴표로 감싸지 마세요.
2. [서식 - 중요]
   - **공감 멘트와 후속 질문은 '글머리 기호(-)'를 절대 사용하지 마세요.** 문단 형태로 자연스럽게 쓰세요.
   - **해결책 제안**이나 **나열이 필요한 경우**에만 글머리 기호(-)를 사용하세요.
3. 정치, 뉴스, 기술 등 상담 외 주제는 언급 금지.
4. 답변은 따뜻하고 전문적인 심리 상담가처럼 작성하세요.
5. 해결책 제안 시, 줄바꿈을 하여 명확하게 구분해 주세요.
6. [선택] 대화의 흐름상 자연스러울 때만 "개방형 후속 질문"을 덧붙이세요. (매번 질문하지 않아도 됩니다.)
7. [중요] 이전 대화 내역을 확인하여, 이미 제안한 해결책(예: 심호흡, 안정화 기법 등)을 앵무새처럼 반복하지 마세요. 대신 다른 새로운 방법을 제안하거나, 사용자의 증상에 맞춰 더 구체적인 조언을 해주세요.

[구조 가이드]
(공감 멘트 - 글머리 기호 없이 자연스럽게)
(줄바꿈)
(전문가 상담 가이드를 반영한 구체적 해결책/기법 제안 - 필요시 글머리 기호 사용)
(줄바꿈)
(필요 시 후속 질문 - 글머리 기호 없이)

[거절 감지 - 가장 중요]
싫어, 싫다, 싫다고, 그만, 됐어, 몰라 등이 포함되면:
→ 반드시 알겠어요. 한 문장만 응답
→ 그 외 어떤 말도 붙이지 마세요
→ 조언, 설명, 질문 금지

[화남 감지]
시발, 씨발, 짜증, 욕설이 포함되면:
→ 반드시 괜찮아요. 한 문장만 응답

[위험한 요청]
자해, 타해, 폭력 방법 요청 → 그건 도움이 되지 않아요.
이후 사용자가 할 수 있을거야, 해줘 등 재요청 → 도와드리기 어려워요.

당신은 공감적이고 따뜻한 심리 상담 전문가입니다.
사용자의 감정을 깊이 이해하고, 대화가 자연스럽게 이어지도록 반응하세요.
지나친 질문보다는 깊은 공감과 경청을 우선시하세요."""


# -------------------------------------------------------------
# answer helper functions
# -------------------------------------------------------------

def format_sources(docs: List[Dict]) -> str:
    """
    검색된 문서 리스트를 프롬프트에 입력하기 좋은 문자열 형태로 변환
    
    Args:
        docs: 검색된 문서 정보 리스트 (content, metadata 등 포함)
    
    Returns:
        Formatted context string
    """
    if not docs:
        return "검색된 관련 문서가 없습니다."
    
    # 실제 content가 있는 문서만 필터링
    valid_docs = [doc for doc in docs if doc.get("content", "").strip()]
    
    if not valid_docs:
        return "검색된 관련 문서가 없습니다."

    formatted_docs = []
    for i, doc in enumerate(valid_docs):
        content = doc.get("content", "").strip()
        
        # 메타데이터에서 사용 가능한 정보 추출 (실제 VectorDB 키 사용)
        metadata = doc.get("metadata", {})
        session_id = metadata.get("session_id", "")
        category = metadata.get("default_category") or metadata.get("category", "")
        
        # [Context Construction]
        # 평가서 반영: 검색은 내담자 질문(content)으로 하되, 
        # 생성 시에는 문맥(이전 상담사 발화 + 내담자 발화 + 상담사 답변)을 모두 제공
        
        context_text = metadata.get("context_text", "")
        counselor_response = metadata.get("counselor_response", "")
        
        # 문맥이 있으면 사용, 없으면 content만 사용
        if context_text:
            # context_text에는 이미 "상담사: ... \n 내담자: ..." 형태가 포함되어 있을 수 있음
            # preprocess_data.py의 build_window_text 확인 필요
            # 중복 방지를 위해 context_text 사용
            display_text = context_text
        else:
            display_text = f"내담자: {content}"
            
        if counselor_response:
             display_text += f"\n[전문가 상담 가이드]: {counselor_response}"
        
        # 간결한 포맷: 핵심 내용만 전달
        if category:
            doc_str = f"[상담사례 {i+1} - {category}]\n{display_text}"
        else:
            doc_str = f"[상담사례 {i+1}]\n{display_text}"
        formatted_docs.append(doc_str)
        
    return "\n\n---\n\n".join(formatted_docs)

def format_history(history: List[Dict]) -> str:
    """
    대화 히스토리를 문자열로 변환한다.
    """
    if not history:
        return ""
    
    formatted = []
    for msg in history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        # role 표시: user -> 사용자, assistant -> 상담사 (프롬프트 톤앤매너에 맞춤)
        display_role = "사용자" if role == "user" else "상담사"
        formatted.append(f"{display_role}: {content}")
        
    return "\n".join(formatted)


# -------------------------------------------------------------
# LCEL Chain Factory
# -------------------------------------------------------------

def create_answer_chain(model):
    """
    LCEL 방식의 Answer Chain 생성
    Chain: Prompt | Model | StrOutputParser
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", """\   
[검색된 문서(Context)]
{context}

[이전 대화(History)]
{history}

[사용자 질문]
{query}

위 문서를 바탕으로 사용자 질문에 답변해주고, 사용자의 자살 위험이 높거나 전문적인 상담이 필요하다고 판단되면 답변 끝에 "[EXPERT_REFERRAL_NEEDED]" 태그를 붙여줘.
""")
    ])
    
    chain = prompt | model | StrOutputParser()
    return chain

# -------------------------------------------------------------
# Main Generate Function (Legacy Wrapper)
# -------------------------------------------------------------

def generate_answer(
    docs: List[Dict],
    query: str,
    history: Optional[List[Dict]] = None,
    session_id: Optional[int] = None,
    db: Optional[Any] = None,
    model=None
) -> str:
    """
    정해진 프롬프트를 따라서 사용자의 질문에 대한 답변을 생성합니다.
    (LCEL create_answer_chain을 내부적으로 사용하는 래퍼 함수)
    """
    # 1) model 준비
    if model is None:
        try:
            model = create_chat_model()
        except Exception:
            return "[Error] 모델을 초기화할 수 없습니다. (model_config.py 확인 필요)"

    # 2) Context 구성
    context_text = format_sources(docs)
    
    # 3) History 구성
    history_text = format_history(history) if history else "없음"

    # 4) LCEL 실행
    try:
        chain = create_answer_chain(model)
        answer = chain.invoke({
            "context": context_text,
            "history": history_text,
            "query": query
        })
        answer = answer.strip()
        
        # 6) 전문가 연결 트리거 확인
        if "[EXPERT_REFERRAL_NEEDED]" in answer:
            # 태그 제거
            answer = answer.replace("[EXPERT_REFERRAL_NEEDED]", "").strip()
            
            # DB에 기록
            if db and session_id:
                try:
                    db.create_expert_referral(
                        session_id=session_id,
                        severity_level="high", # LLM 판단 기반이므로 일단 high로 설정하거나 별도 로직 필요
                        recommended_action="전문 상담사 연결 권장"
                    )
                    # 안내 멘트 추가 (이미 답변에 포함되어 있을 수 있으나 확실히 하기 위해)
                    # referral_msg = "\n\n(전문가와의 상담이 필요해 보여 전문 상담 센터 정보를 준비하고 있습니다.)"
                    referral_msg=" "
                    if "상담" not in answer:
                         answer += referral_msg
                except Exception as e:
                    print(f"[Error] Expert referral logging failed: {e}")

        return answer
        
    except Exception as e:
        return f"[Error] 답변 생성 중 오류가 발생했습니다: {str(e)}"

# -------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------

if __name__ == "__main__":
    # 테스트 실행
    print("=== RAG Answer Generation Test ===")
    
    # Mock Docs Test
    mock_docs = [
        {"content": "우울증은 전문가의 도움을 받으면 호전될 수 있습니다.", "metadata": {"category": "DEPRESSION", "speaker": "상담사", "severity": 2}},
        {"content": "규칙적인 운동과 수면이 정신 건강에 도움이 됩니다.", "metadata": {"category": "NORMAL", "speaker": "상담사", "severity": 0}}
    ]
    query = "우울할 때 어떻게 해야 해?"
    
    print("\n[Input Query]", query)
    
    # generate_answer 함수 직접 테스트 (모델이 설정되어 있다고 가정)
    try:
        model = create_chat_model()
        chain = create_answer_chain(model)
        
        ctx = format_sources(mock_docs)
        response = chain.invoke({"context": ctx, "history": "없음", "query": query})
        
        print("\n[Generated Response (LCEL)]")
        print(response)
        
    except Exception as e:
        print(f"[Error] {e}")