"""
FileName    : chain.py
Auth        : 우재현
Date        : 2026-01-29
Description : RAG 전체 파이프라인 관리
Issue/Note  : DB 연결, Rewrite, Retrieve, Answer 모든 단계 통합
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Root 경로 설정
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from config.model_config import create_chat_model
from src.database.db_manager import DatabaseManager
from src.database.vector_store import VectorStore
from src.rag.retriever import create_retriever, load_vector_db
from src.rag.rewrite import create_rewrite_chain, format_history
from src.rag.answer import create_answer_chain, format_sources
from src.rag.intent_router import QueryIntent, route_query, CRISIS_INTENTS

# -------------------------------------------------------------
# RAG Main Class
# -------------------------------------------------------------

class RAGChain:
    """
    RAG 시스템의 전체 흐름을 제어하는 클래스 (LCEL 기반 전체 파이프라인 구성)
    """
    
    def __init__(self, db_manager: DatabaseManager = None):
        """
        초기화 및 체인 구성
        """
        # 1. DB Manager
        self.db = db_manager if db_manager else DatabaseManager()
        
        # 2. Vector DB 로드
        self.vector_db = load_vector_db()
        
        # 3. 모델 및 컴포넌트 초기화
        self.model = create_chat_model()
        retriever_func = create_retriever(self.vector_db)
        
        # 4. 서브 체인 정의
        rewrite_chain = create_rewrite_chain(self.model)
        answer_chain = create_answer_chain(self.model)
        
        # ---------------------------------------------------------
        # RAG 전체 파이프라인 구성 (Compose Full RAG Pipeline)
        # Input: {"query": str, "history_text": str}
        # ---------------------------------------------------------
        
        # 1. 질문 재작성 (Rewrite)
        # Input: {query, history_text} -> Output: rewritten_query (str)
        rewrite_step = RunnablePassthrough.assign(
            rewritten_query=lambda x: rewrite_chain.invoke({
                "history": x["history_text"], 
                "query": x["query"]
            }).strip().strip('"\'').splitlines()[0]
        )
        
        # 2. 문서 검색 (Retrieve) & Context 포맷팅
        # Input: {..., rewritten_query} -> Output: source_docs (List), context (Str)
        def retrieve_and_format(x):
            docs = retriever_func(query=x["rewritten_query"])
            
            is_low_similarity = False
            
            # [유사도 점수 기반 경고]
            # 유사도(Similarity) <= 0.65 인 경우 경고 (Distance >= 0.65)
            if docs:
                first_dist = docs[0].get("distance")
                if first_dist is not None and first_dist >= 0.65:
                    print("[Warning] 데이터 내에 유사한 정보가 없어서 임의의 내용을 출력 중입니다.")
                    is_low_similarity = True
            else:
                 print("[Warning] 데이터 내에 유사한 정보가 없어서 임의의 내용을 출력 중입니다.")
                 is_low_similarity = True

            for i, doc in enumerate(docs):
                dist = doc.get("distance")
                if dist is not None:
                    print(f"[Retrieval] Doc {i+1} distance: {dist:.4f}")
                else:
                    print(f"[Retrieval] Doc {i+1} distance: None")
            return {"source_docs": docs, "context": format_sources(docs), "is_low_similarity": is_low_similarity}

        retrieve_step = RunnablePassthrough.assign(
            **{
                "data": lambda x: retrieve_and_format(x)
            }
        ) | RunnablePassthrough.assign(
            source_docs=lambda x: x["data"]["source_docs"],
            context=lambda x: x["data"]["context"],
            is_low_similarity=lambda x: x["data"]["is_low_similarity"]
        )
        
        # 3. 답변 생성 (Answer)
        # Input: {..., context, history_text, rewritten_query} -> Output: answer (최종 답변)
        def conditional_answer(x):
            if x.get("is_low_similarity", False):
                return "해당 질문에는 답변을 드리기 어렵습니다. 다른 질문을 부탁드립니다."
            return answer_chain.invoke({
                "context": x["context"],
                "history": x["history_text"],
                "query": x["rewritten_query"] 
            })
            
        answer_step = RunnablePassthrough.assign(
            answer=lambda x: conditional_answer(x)
        )
        
        # Pipeline Chain
        self.rag_pipeline = rewrite_step | retrieve_step | answer_step

    def run(self, user_id: int, session_id: int, query: str) -> str:
        """
        사용자 발화에 대한 RAG 응답 생성 및 처리 전체 과정
        Intent Router를 통해 처리 경로 분기
        """
        print(f"\n[Flow Start] User: {user_id}, Session: {session_id}")
        
        # 1. 사용자 메시지 저장
        self.db.add_chat_message(session_id, "user", query)
        
        # 2. 대화 히스토리 로드
        history_objs = self.db.get_chat_history(session_id)
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in history_objs]
        pre_history = history_dicts[:-1]
        history_text = format_history(pre_history)
        
        print(f"[Step] Input: {query}")
        
        start_time = time.time()
        
        try:
            # 3. Intent 분류 및 라우팅
            intent, direct_response, needs_rag = route_query(query, self.model)
            print(f"[Router] Intent: {intent.value}, Needs RAG: {needs_rag}")
            
            # 4. 분기 처리
            if direct_response and not needs_rag:
                # 직접 응답 (GREETING, CHITCHAT, CRISIS)
                answer = direct_response
                
                # CRISIS인 경우 전문가 연결 기록
                if intent in CRISIS_INTENTS:
                    self._handle_expert_referral(session_id, answer)
            else:
                # RAG 파이프라인 실행 (EMOTION, QUESTION)
                result = self.rag_pipeline.invoke({
                    "query": query,
                    "history_text": history_text
                })
                answer = result["answer"].strip()
            
            # 5. 전문가 연결 태그 후처리
            if "[EXPERT_REFERRAL_NEEDED]" in answer:
                answer = answer.replace("[EXPERT_REFERRAL_NEEDED]", "").strip()
                self._handle_expert_referral(session_id, answer)
            
            # 6. Assistant 메시지 저장
            self.db.add_chat_message(session_id, "assistant", answer)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"[System] Response Time: {elapsed_time:.2f}s")
            
            print(f"[Flow End] 답변 생성 완료 (Intent: {intent.value})")
            return answer
            
        except Exception as e:
            print(f"[Error] RAG 파이프라인 실패: {e}")
            return "죄송합니다. 처리 중 오류가 발생했습니다."

    def stream(self, user_id: int, session_id: int, query: str, debug: bool = False):
        """
        스트리밍 방식으로 RAG 응답 생성 (SSE용 제너레이터)
        
        Args:
            user_id: 사용자 ID
            session_id: 세션 ID
            query: 사용자 질문
            debug: 디버그 모드 여부
            
        Yields:
            str or dict: 텍스트 청크 또는 디버그 정보
        """
        print(f"\n[Stream Start] User: {user_id}, Session: {session_id}")
        
        # 1. 사용자 메시지 저장
        self.db.add_chat_message(session_id, "user", query)
        
        # 2. 대화 히스토리 로드
        history_objs = self.db.get_chat_history(session_id)
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in history_objs]
        pre_history = history_dicts[:-1]
        history_text = format_history(pre_history)
        
        try:
            # 3. Intent 분류 및 라우팅
            intent, direct_response, needs_rag = route_query(query, self.model)
            print(f"[Router] Intent: {intent.value}, Needs RAG: {needs_rag}")
            
            # 4. 분기 처리
            if direct_response and not needs_rag:
                # 직접 응답
                answer = direct_response
                rewritten_query = query  # 직접 응답은 rewrite 없음
                source_docs = []
                
                if intent in CRISIS_INTENTS:
                    self._handle_expert_referral(session_id, answer)
            else:
                # RAG 파이프라인 실행
                result = self.rag_pipeline.invoke({
                    "query": query,
                    "history_text": history_text
                })
                
                answer = result["answer"].strip()
                rewritten_query = result.get("rewritten_query", "")
                source_docs = result.get("source_docs", [])
                
                # 전문가 연결 태그 처리
                if "[EXPERT_REFERRAL_NEEDED]" in answer:
                    answer = answer.replace("[EXPERT_REFERRAL_NEEDED]", "").strip()
                    self._handle_expert_referral(session_id, answer)
            
            # 5. 스트리밍 출력
            if debug:
                # 디버그 모드: 상세 디버그 정보 전송
                
                # 소스 문서 상세 정보 정리
                sources_detail = []
                for i, doc in enumerate(source_docs[:5]):
                    meta = doc.get("metadata", {})
                    content = doc.get("content", "")
                    # speaker 정보 변환 (counselor -> 상담사, client -> 내담자)
                    raw_speaker = meta.get("speaker", "")
                    if raw_speaker == "counselor":
                        speaker = "상담사"
                    elif raw_speaker == "client":
                        speaker = "내담자"
                    else:
                        speaker = raw_speaker or "N/A"
                    
                    sources_detail.append({
                        "rank": i + 1,
                        "session_id": meta.get("session_id", "N/A"),
                        "category": meta.get("category", "N/A"),
                        "turn_idx": meta.get("turn_idx", "N/A"),
                        "speaker": speaker,
                        "distance": round(doc.get("distance", 0), 4),
                        "content": content[:200] + "..." if len(content) > 200 else content
                    })
                
                # context 길이 계산
                try:
                    context_text = result.get("context", "")
                except NameError:
                    context_text = ""
                context_length = len(context_text)
                
                # 통합 디버그 정보 전송
                yield {
                    "type": "debug",
                    "data": {
                        "intent": intent.value,
                        "rewritten_query": rewritten_query,
                        "context_length": context_length,
                        "sources_count": len(source_docs),
                        "sources": sources_detail
                    }
                }
                # 텍스트는 별도로 전송 (app.js에서 'content' 타입으로 처리)
                yield {"type": "content", "data": answer}
            else:
                # 일반 모드: 문자 단위 스트리밍
                for char in answer:
                    yield char
            
            # 6. Assistant 메시지 저장
            self.db.add_chat_message(session_id, "assistant", answer)
            
            print(f"[Stream End] 완료 (Intent: {intent.value})")
            
        except Exception as e:
            print(f"[Error] 스트리밍 파이프라인 실패: {e}")
            error_msg = "죄송합니다. 처리 중 오류가 발생했습니다."
            if debug:
                yield {"type": "error", "data": str(e)}
                yield {"type": "content", "data": error_msg}
            else:
                for char in error_msg:
                    yield char

    def run_with_debug(self, query: str, history: List[Dict[str, str]] = []) -> Dict[str, Any]:
        """
        [테스트/디버깅용] RAG 파이프라인의 중간 결과까지 포함하여 반환합니다.
        
        Args:
            query: 사용자 질문
            history: 대화 프롬프트용 히스토리 리스트 [{"role": "user", "content": "..."}]
        
        Returns:
            Dict: {
                "input_query": str,
                "rewritten_query": str,
                "source_docs": List[Dict], # {content, metadata, distance}
                "context": str,
                "answer": str
            }
        """
        history_text = format_history(history)
        
        try:
            result = self.rag_pipeline.invoke({
                "query": query,
                "history_text": history_text
            })
            
            # 후처리 전 순수 답변 반환
            return {
                "input_query": query,
                "rewritten_query": result.get("rewritten_query", ""),
                "source_docs": result.get("source_docs", []),
                "context": result.get("context", ""),
                "answer": result.get("answer", "").strip()
            }
            
        except Exception as e:
            return {
                "error": str(e)
            }

    def _handle_expert_referral(self, session_id: int, answer: str):
        """전문가 연결 DB 기록"""
        try:
            self.db.create_expert_referral(
                session_id=session_id,
                severity_level="high",
                recommended_action="전문 상담사 연결 권장"
            )
        except Exception as e:
            print(f"[Error] 전문가 연결 로깅 실패: {e}")

# -------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    # Test Setup
    print("=== RAG Chain Test (LCEL) ===")
    
    # 임시 DB Manager (테스트용)
    test_db = DatabaseManager(echo=False)
    rag_chain = RAGChain(db_manager=test_db)
    
    # 1. User/Session Create
    try:
        user = test_db.create_user("test_lcel_user_01")
    except Exception:
        test_db.session.rollback()  # IntegrityError 후 rollback 필요
        user = test_db.get_user_by_username("test_lcel_user_01")
        
    session = test_db.create_chat_session(user.id)
    
    # 2. Run Flow
    q1 = "사는게 재미가 없어"
    ans1 = rag_chain.run(user.id, session.id, q1)
    print(f"\n[Bot]: {ans1}\n")
    
    # 3. 연속 대화 - 맥락 인식 테스트
    print("=" * 50)
    q2 = "그래서 어떻게 해야 해?"
    ans2 = rag_chain.run(user.id, session.id, q2)
    print(f"\n[Bot]: {ans2}\n")
