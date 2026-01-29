"""
FileName    : chain.py
Auth        : 우재현
Date        : 2026-01-06
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
from src.rag.intent_router import route_query, QueryIntent, should_use_rag

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
        [2026-01-28] Intent Router 통합 - 의도에 따라 RAG 실행 여부 결정
        """
        print(f"\n[Flow Start] User: {user_id}, Session: {session_id}")
        
        # 1. 사용자 메시지 저장
        self.db.add_chat_message(session_id, "user", query)
        
        print(f"[Step] Input: {query}")
        
        start_time = time.time()
        
        try:
            # ============================================
            # 2. Intent Router: 의도 분류
            # ============================================
            intent, direct_response, needs_rag = route_query(query, self.model)
            print(f"[Intent] {intent.value} | Needs RAG: {needs_rag}")
            
            # ============================================
            # 3-A. GREETING/CHITCHAT: RAG 없이 직접 응답
            # ============================================
            if not needs_rag and direct_response:
                # CLOSING 인 경우 요약 생성 시도 (만약 direct_response가 단순 인사라면 덮어쓰기 될 수 있음, 
                # 하지만 intent_router에서 CLOSING은 direct_response가 없을 수도 있음. 확인 필요. 
                # 일단 CLOSING은 별도 로직으로 처리)
                pass

            if intent == QueryIntent.CLOSING:
                answer = self._generate_session_summary(session_id)
                self.db.add_chat_message(session_id, "assistant", answer)
                print(f"[Flow End] 상담 요약 및 종료 완료")
                return answer

            if not needs_rag and direct_response:
                answer = direct_response
                print(f"[Direct Response] Intent={intent.value}")
                
                # 5. Assistant 메시지 저장
                self.db.add_chat_message(session_id, "assistant", answer)
                
                end_time = time.time()
                print(f"[System] Response Time: {end_time - start_time:.2f}s")
                print(f"[Flow End] 직접 응답 완료")
                return answer
            
            # ============================================
            # 3-B. CRISIS: 긴급 응답 + 전문가 연결
            # ============================================
            if intent == QueryIntent.CRISIS:
                answer = direct_response if direct_response else "지금 많이 힘드시군요. 전문 상담사와 이야기해 보시는 것을 권해드려요. 📞 자살예방상담전화: 1393 (24시간)"
                self._handle_expert_referral(session_id, answer)
                self.db.add_chat_message(session_id, "assistant", answer)
                
                end_time = time.time()
                print(f"[System] Response Time: {end_time - start_time:.2f}s")
                print(f"[Flow End] 위기 대응 완료")
                return answer
            
            # ============================================
            # 3-C. EMOTION/QUESTION: RAG 파이프라인 실행
            # ============================================
            
            # 대화 히스토리 로드
            history_objs = self.db.get_chat_history(session_id)
            history_dicts = [{"role": msg.role, "content": msg.content} for msg in history_objs]
            pre_history = history_dicts[:-1]
            history_text = format_history(pre_history)
            
            # RAG 파이프라인 실행
            result = self.rag_pipeline.invoke({
                "query": query,
                "history_text": history_text
            })
            
            answer = result["answer"].strip()
            
            # 4. 전문가 연결 감지 (후처리)
            if "[EXPERT_REFERRAL_NEEDED]" in answer:
                answer = answer.replace("[EXPERT_REFERRAL_NEEDED]", "").strip()
                self._handle_expert_referral(session_id, answer)
                if "상담" not in answer:
                    answer += "\n"
            
            # 5. Assistant 메시지 저장
            self.db.add_chat_message(session_id, "assistant", answer)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"[System] Response Time: {elapsed_time:.2f}s")
            
            print(f"[Flow End] RAG 응답 완료")
            return answer
            
        except Exception as e:
            print(f"[Error] RAG 파이프라인 실패: {e}")
            return "죄송합니다. 처리 중 오류가 발생했습니다."

    def stream(self, user_id: int, session_id: int, query: str, debug: bool = False):
        """
        사용자 발화에 대한 RAG 응답을 스트리밍으로 반환
        debug=True 일 경우: 
            yield {"type": "debug", "data": {...}}  # 검색 결과 등
            yield {"type": "content", "data": "..."} # 답변 청크
        debug=False 일 경우:
            yield "답변 청크" ...
        """
        print(f"\n[Flow Start (Stream)] User: {user_id}, Session: {session_id}, Debug: {debug}")
        
        # 1. 사용자 메시지 저장
        self.db.add_chat_message(session_id, "user", query)
        
        try:
            # 2. Intent Router
            intent, direct_response, needs_rag = route_query(query, self.model)
            
            # 3-A. GREETING/CHITCHAT: 직접 응답
            if not needs_rag and direct_response:
                # Debug Info Yield
                if debug:
                    yield {
                        "type": "debug",
                        "data": {
                            "intent": intent.value,
                            "rewritten_query": query,
                            "sources": [],
                            "context_length": 0,
                            "note": "RAG 검색 없이 직접 응답 (Intent Router)"
                        }
                    }

                self.db.add_chat_message(session_id, "assistant", direct_response)
                # 스트리밍 흉내 (자연스러운 타이핑 효과)
                for i in range(0, len(direct_response), 3): # 3글자씩 끊어서
                    chunk = direct_response[i:i+3]
                    if debug:
                        yield {"type": "content", "data": chunk}
                    else:
                        yield chunk
                    time.sleep(0.05)
                return

            # 3-B. CRISIS: 위기 대응
            if intent == QueryIntent.CRISIS:
                answer = direct_response if direct_response else "지금 많이 힘드시군요. 전문 상담사와 이야기해 보시는 것을 권해드려요. 📞 자살예방상담전화: 1393 (24시간)"
                
                # Debug Info Yield
                if debug:
                    yield {
                        "type": "debug",
                        "data": {
                            "intent": intent.value,
                            "rewritten_query": query,
                            "sources": [],
                            "context_length": 0,
                            "note": "위기 상황 - 즉시 전문가 연결"
                        }
                    }

                self._handle_expert_referral(session_id, answer)
                self.db.add_chat_message(session_id, "assistant", answer)
                
                # 스트리밍 흉내
                for i in range(0, len(answer), 3):
                    chunk = answer[i:i+3]
                    if debug:
                        yield {"type": "content", "data": chunk}
                    else:
                        yield chunk
                    time.sleep(0.05)
                return
            
            # 3-C. RAG 파이프라인 실행
            
            # History Load
            history_objs = self.db.get_chat_history(session_id)
            history_dicts = [{"role": msg.role, "content": msg.content} for msg in history_objs]
            # 마지막 사용자 메시지는 이미 저장했으므로, 그 이전까지를 히스토리로 사용
            pre_history = history_dicts[:-1]
            history_text = format_history(pre_history)
            
            # Rewrite (Sync)
            rewrite_chain = create_rewrite_chain(self.model)
            rewritten_query = rewrite_chain.invoke({
                "history": history_text, 
                "query": query
            }).strip().strip('"\'').splitlines()[0]
            
            # Retrieve (Sync)
            retriever_func = create_retriever(self.vector_db)
            docs = retriever_func(query=rewritten_query)
            
            # [Relevance Filtering]
            SIMILARITY_THRESHOLD = 0.40 # 한국어 임베딩 모델(ko-sroberta) 거리 척도에 맞춤
            valid_docs = []
            
            # Debug Info Formatting & Yield
            if debug:
                debug_info_sources = []
                for i, doc in enumerate(docs[:5]):
                    meta = doc.get("metadata", {})
                    distance = round(doc.get("distance", meta.get("distance", 0)), 4)
                    
                    is_valid = distance <= SIMILARITY_THRESHOLD
                    if is_valid:
                        valid_docs.append(doc)
                    
                    window_text = meta.get("window_text", "") or ""
                    content = doc.get("content", "")
                    display_content = window_text if len(window_text) > len(content) else content
                    
                    status_prefix = "" if is_valid else "[SKIPPED-Low Relevance] "
                    
                    debug_info_sources.append({
                        "rank": i + 1,
                        "session_id": meta.get("session_id", "N/A"),
                        "category": meta.get("category", "N/A"),
                        "turn_idx": meta.get("turn_idx", "N/A"),
                        "content": status_prefix + f"내담자: {display_content}\n[상담사 답변]: {meta.get('counselor_response', '(답변 없음)')[:100]}...",
                        "distance": distance
                    })

                yield {
                    "type": "debug",
                    "data": {
                        "intent": intent.value,
                        "rewritten_query": rewritten_query,
                        "sources": debug_info_sources,
                        "context_length": sum(len(d.get("page_content", "") or d.get("content", "")) for d in valid_docs),
                        "note": f"Threshold({SIMILARITY_THRESHOLD}) 적용: {len(valid_docs)}/{len(docs)} 건 사용"
                    }
                }
                
                # Update docs to only valid ones for context generation
                docs = valid_docs
            else:
                # Debug mode 아닐 때도 필터링 적용
                docs = [d for d in docs if d.get("distance", d.get("metadata", {}).get("distance", 0)) <= SIMILARITY_THRESHOLD]

            if not docs:
                context = "관련된 상담 내역이 없습니다. (위로와 공감, 일반적인 심리학 지식에 기반하여 답변하세요)"
            else:
                context = format_sources(docs)
            
            # 유사도 체크 (옵션)
            if not docs:
                pass 
            
            # Answer Stream
            answer_chain = create_answer_chain(self.model)
            full_answer = ""
            
            for chunk in answer_chain.stream({
                "context": context,
                "history": history_text,
                "query": rewritten_query
            }):
                full_answer += chunk
                if debug:
                    yield {"type": "content", "data": chunk}
                else:
                    yield chunk
            
            # 4. 전문가 연결 감지 및 후처리 (Logging Only)
            clean_answer = full_answer
            if "[EXPERT_REFERRAL_NEEDED]" in full_answer:
                clean_answer = full_answer.replace("[EXPERT_REFERRAL_NEEDED]", "").strip()
                self._handle_expert_referral(session_id, clean_answer)
                if "상담" not in clean_answer:
                    clean_answer += "\n"
            
            # 5. Assistant 메시지 저장 (Cleaned version)
            self.db.add_chat_message(session_id, "assistant", clean_answer)
            print(f"[Flow End (Stream)] RAG 응답 완료")
            
        except Exception as e:
            print(f"[Error] 스트리밍 중 오류: {e}")
            err_msg = "죄송합니다. 처리 중 오류가 발생했습니다."
            if debug:
                yield {"type": "content", "data": err_msg}
            else:
                yield err_msg

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

    def _generate_session_summary(self, session_id: int) -> str:
        """
        [2026-01-29] 상담 종료 시, 이번 세션에서 제안된 안정화 기법/조언 요약
        """
        history = self.db.get_chat_history(session_id)
        if not history:
            return "진행된 상담 내역이 없어 요약할 내용이 없습니다. 언제든 다시 찾아주세요."

        conversation_text = ""
        for msg in history:
            if msg.role == "system": continue
            role = "상담사" if msg.role == "assistant" else "내담자"
            conversation_text += f"{role}: {msg.content}\n"

        summary_prompt = f"""
[역할]
당신은 심리 상담 내용을 정리해주는 AI 비서입니다.
아래 대화 기록을 바탕으로, **상담사가 내담자에게 제안했던 심리적 안정화 기법이나 실질적인 조언들**을 요약해서 정리해 주세요.

[대화 기록]
{conversation_text}

[요약 규칙]
1. 상담사가 제안한 **구체적인 해결책, 기법(예: 호흡법, 점수 매기기 등), 행동 지침**만 추출하세요.
2. 단순히 "공감해주었다" 같은 내용은 적지 마세요.
3. 내담자가 실천할 수 있도록 [오늘의 심리 처방] 형식으로 깔끔하게 리스트업 해주세요.
4. 마지막에는 따뜻한 격려의 한 마디로 마무리하세요.

[출력 형식]
[오늘의 심리 처방] 📝
1. (기법 이름): (구체적 방법 요약)
2. ...

(마무리 격려)
"""
        response = self.model.invoke(summary_prompt)
        return response.content if hasattr(response, 'content') else str(response)

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
        user = test_db.get_user_by_username("test_lcel_user_01")
        
    session = test_db.create_chat_session(user.id)
    
    # 2. Run Flow
    q1 = "사는게 재미가 없어"
    ans1 = rag_chain.run(user.id, session.id, q1)
    print(f"\n[Bot]: {ans1}\n")

