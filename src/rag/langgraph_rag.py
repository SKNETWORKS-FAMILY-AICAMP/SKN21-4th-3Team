"""
FileName    : langgraph_rag.py
Auth        : 우재현
Date        : 2026-01-29
Description : LangGraph 기반 RAG 파이프라인
              StateGraph를 활용한 그래프 기반 흐름 제어
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

import sys
import time
import re
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langgraph.graph import StateGraph, END
from config.model_config import create_chat_model
from src.database.db_manager import DatabaseManager
from src.database.vector_store import VectorStore
from src.rag.retriever import create_retriever, load_vector_db
from src.rag.rewrite import create_rewrite_chain, format_history
from src.rag.answer import create_answer_chain, format_sources
from src.rag.intent_router import route_query, QueryIntent

# -------------------------------------------------------------
# 특수 토큰 필터링
# -------------------------------------------------------------

SPECIAL_TOKEN_PATTERN = re.compile(
    r'<\|[^|>]+\|>'
    r'|\.stdin|\.stdout'
)

def filter_special_tokens(text: str) -> str:
    if not text:
        return text
    return SPECIAL_TOKEN_PATTERN.sub('', text)

# -------------------------------------------------------------
# State 정의
# -------------------------------------------------------------

class RAGState(TypedDict):
    """LangGraph 상태 정의"""
    # 입력
    query: str
    user_id: int
    session_id: int
    
    # 대화 컨텍스트
    history_text: str
    
    # Intent 분류 결과
    intent: str
    needs_rag: bool
    direct_response: str
    
    # RAG 파이프라인 결과
    rewritten_query: str
    source_docs: List[Dict]
    context: str
    is_low_similarity: bool
    
    # 최종 출력
    answer: str

# -------------------------------------------------------------
# LangGraphRAG 클래스
# -------------------------------------------------------------

class LangGraphRAG:
    """
    LangGraph 기반 RAG 시스템
    
    그래프 구조:
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │   [START] → [classify_intent]                            │
    │                    │                                     │
    │         ┌─────────┴─────────┐                            │
    │         ↓                   ↓                            │
    │   needs_rag=False     needs_rag=True                     │
    │         │                   │                            │
    │         ↓                   ↓                            │
    │   [direct_respond]    [rewrite] → [retrieve] → [answer]  │
    │         │                                          │     │
    │         └──────────────────┬───────────────────────┘     │
    │                            ↓                             │
    │                          [END]                           │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, db_manager: DatabaseManager = None):
        print("[LangGraphRAG] 초기화 시작")
        
        # DB 및 모델 초기화
        self.db = db_manager if db_manager else DatabaseManager()
        self.vector_db = load_vector_db()
        self.model = create_chat_model()
        self.retriever_func = create_retriever(self.vector_db)
        
        # 체인 생성
        self.rewrite_chain = create_rewrite_chain(self.model)
        self.answer_chain = create_answer_chain(self.model)
        
        # 그래프 컴파일
        self.graph = self._build_graph()
        
        print("[LangGraphRAG] 초기화 완료")
    
    # ---------------------------------------------------------
    # 그래프 빌드
    # ---------------------------------------------------------
    
    def _build_graph(self) -> StateGraph:
        """StateGraph 생성 및 컴파일"""
        
        workflow = StateGraph(RAGState)
        
        # 노드 등록
        workflow.add_node("classify_intent", self._node_classify_intent)
        workflow.add_node("direct_respond", self._node_direct_respond)
        workflow.add_node("rewrite", self._node_rewrite)
        workflow.add_node("retrieve", self._node_retrieve)
        workflow.add_node("answer", self._node_answer)
        
        # 진입점
        workflow.set_entry_point("classify_intent")
        
        # 조건부 라우팅: Intent에 따라 분기
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "direct": "direct_respond",
                "rag": "rewrite"
            }
        )
        
        # RAG 파이프라인 순차 연결
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("retrieve", "answer")
        
        # 종료 연결
        workflow.add_edge("direct_respond", END)
        workflow.add_edge("answer", END)
        
        return workflow.compile()
    
    # ---------------------------------------------------------
    # 라우팅 함수
    # ---------------------------------------------------------
    
    def _route_by_intent(self, state: RAGState) -> str:
        """Intent에 따라 분기 결정"""
        if state.get("needs_rag", True):
            return "rag"
        return "direct"
    
    # ---------------------------------------------------------
    # 노드 함수들
    # ---------------------------------------------------------
    
    def _node_classify_intent(self, state: RAGState) -> dict:
        """[Node] Intent 분류"""
        query = state["query"]
        print(f"[LangGraph] Node: classify_intent | Query: {query}")
        
        intent, direct_response, needs_rag = route_query(query, self.model)
        print(f"[LangGraph] Intent: {intent.value} | Needs RAG: {needs_rag}")
        
        return {
            "intent": intent.value,
            "needs_rag": needs_rag,
            "direct_response": direct_response or ""
        }
    
    def _node_direct_respond(self, state: RAGState) -> dict:
        """[Node] 직접 응답 (RAG 불필요)"""
        intent = state.get("intent", "")
        direct_response = state.get("direct_response", "")
        
        print(f"[LangGraph] Node: direct_respond | Intent: {intent}")
        
        # CLOSING 처리: 세션 요약 생성
        if intent == QueryIntent.CLOSING.value:
            answer = self._generate_session_summary(state["session_id"])
        else:
            answer = direct_response
        
        return {"answer": answer}
    
    def _node_rewrite(self, state: RAGState) -> dict:
        """[Node] 쿼리 재작성"""
        query = state["query"]
        history_text = state.get("history_text", "없음")
        
        print(f"[LangGraph] Node: rewrite")
        
        rewritten = self.rewrite_chain.invoke({
            "history": history_text,
            "query": query
        }).strip().strip('"\'').splitlines()[0]
        
        print(f"[LangGraph] Rewritten: {rewritten}")
        
        return {"rewritten_query": rewritten}
    
    def _node_retrieve(self, state: RAGState) -> dict:
        """[Node] 문서 검색"""
        rewritten_query = state["rewritten_query"]
        
        print(f"[LangGraph] Node: retrieve")
        
        docs = self.retriever_func(query=rewritten_query)
        
        # 유사도 체크
        is_low_similarity = False
        if docs:
            first_dist = docs[0].get("distance")
            if first_dist is not None and first_dist >= 0.65:
                print("[LangGraph] Warning: 유사도 낮음")
                is_low_similarity = True
        else:
            print("[LangGraph] Warning: 검색 결과 없음")
            is_low_similarity = True
        
        # 거리 로깅
        for i, doc in enumerate(docs):
            dist = doc.get("distance")
            print(f"[LangGraph] Doc {i+1} distance: {dist:.4f}" if dist else f"[LangGraph] Doc {i+1} distance: None")
        
        context = format_sources(docs)
        
        return {
            "source_docs": docs,
            "context": context,
            "is_low_similarity": is_low_similarity
        }
    
    def _node_answer(self, state: RAGState) -> dict:
        """[Node] 답변 생성"""
        print(f"[LangGraph] Node: answer")
        
        # 유사도 낮으면 기본 응답
        if state.get("is_low_similarity", False):
            return {"answer": "해당 질문에는 답변을 드리기 어렵습니다. 다른 질문을 부탁드립니다."}
        
        answer = self.answer_chain.invoke({
            "context": state["context"],
            "history": state.get("history_text", "없음"),
            "query": state["rewritten_query"]
        })
        
        # 후처리
        answer = filter_special_tokens(answer).strip()
        
        # 전문가 연결 태그 처리
        if "[EXPERT_REFERRAL_NEEDED]" in answer:
            answer = answer.replace("[EXPERT_REFERRAL_NEEDED]", "").strip()
            self._handle_expert_referral(state["session_id"], answer)
        
        return {"answer": answer}
    
    # ---------------------------------------------------------
    # Public Methods
    # ---------------------------------------------------------
    
    def run(self, user_id: int, session_id: int, query: str) -> str:
        """
        동기 실행 (기존 RAGChain.run 호환)
        """
        print(f"\n[LangGraphRAG] Start | User: {user_id}, Session: {session_id}")
        
        # 사용자 메시지 저장
        self.db.add_chat_message(session_id, "user", query)
        
        start_time = time.time()
        
        try:
            # 히스토리 로드
            history_objs = self.db.get_chat_history(session_id)
            history_dicts = [{"role": msg.role, "content": msg.content} for msg in history_objs]
            pre_history = history_dicts[:-1]
            history_text = format_history(pre_history)
            
            # 그래프 실행
            initial_state: RAGState = {
                "query": query,
                "user_id": user_id,
                "session_id": session_id,
                "history_text": history_text,
                "intent": "",
                "needs_rag": True,
                "direct_response": "",
                "rewritten_query": "",
                "source_docs": [],
                "context": "",
                "is_low_similarity": False,
                "answer": ""
            }
            
            result = self.graph.invoke(initial_state)
            answer = result["answer"]
            
            # Assistant 메시지 저장
            self.db.add_chat_message(session_id, "assistant", answer)
            
            elapsed = time.time() - start_time
            print(f"[LangGraphRAG] End | Time: {elapsed:.2f}s")
            
            return answer
            
        except Exception as e:
            print(f"[LangGraphRAG] Error: {e}")
            return "죄송합니다. 처리 중 오류가 발생했습니다."
    
    def stream(self, user_id: int, session_id: int, query: str, debug: bool = False):
        """
        스트리밍 실행 (기존 RAGChain.stream 호환)
        """
        print(f"\n[LangGraphRAG] Stream Start | User: {user_id}, Session: {session_id}, Debug: {debug}")
        
        # 사용자 메시지 저장
        self.db.add_chat_message(session_id, "user", query)
        
        try:
            # [FIX] 히스토리를 먼저 로드하여 Intent 분류 및 RAG 파이프라인에서 사용
            history_objs = self.db.get_chat_history(session_id)
            history_dicts = [{"role": msg.role, "content": msg.content} for msg in history_objs]
            # 마지막 사용자 메시지는 이미 저장했으므로, 그 이전까지를 히스토리로 사용
            pre_history = history_dicts[:-1] if len(history_dicts) > 1 else []
            history_text = format_history(pre_history)
            
            print(f"[LangGraphRAG] History loaded: {len(pre_history)} messages")
            
            # Intent 분류 (히스토리 컨텍스트와 함께)
            intent, direct_response, needs_rag = route_query(query, self.model, history=pre_history)
            
            # 직접 응답 (RAG 불필요)
            if not needs_rag and direct_response:
                if debug:
                    yield {
                        "type": "debug",
                        "data": {
                            "intent": intent.value,
                            "rewritten_query": query,
                            "sources": [],
                            "context_length": 0,
                            "note": "Intent Router 직접 응답"
                        }
                    }
                
                self.db.add_chat_message(session_id, "assistant", direct_response)
                
                for i in range(0, len(direct_response), 3):
                    chunk = direct_response[i:i+3]
                    if debug:
                        yield {"type": "content", "data": chunk}
                    else:
                        yield chunk
                    time.sleep(0.05)
                return
            
            # RAG 파이프라인 (history_text는 위에서 이미 로드됨)
            
            # Rewrite
            rewritten_query = self.rewrite_chain.invoke({
                "history": history_text,
                "query": query
            }).strip().strip('"\'').splitlines()[0]
            
            # Retrieve
            docs = self.retriever_func(query=rewritten_query)
            
            # 유사도 필터링
            SIMILARITY_THRESHOLD = 0.40
            valid_docs = [d for d in docs if d.get("distance", 1.0) <= SIMILARITY_THRESHOLD]
            
            if debug:
                debug_sources = []
                for i, doc in enumerate(docs[:5]):
                    meta = doc.get("metadata", {})
                    distance = round(doc.get("distance", 0), 4)
                    is_valid = distance <= SIMILARITY_THRESHOLD
                    if is_valid and doc not in valid_docs:
                        valid_docs.append(doc)
                    
                    debug_sources.append({
                        "rank": i + 1,
                        "session_id": meta.get("session_id", "N/A"),
                        "distance": distance,
                        "content": f"{'[SKIPPED] ' if not is_valid else ''}{doc.get('content', '')[:100]}..."
                    })
                
                yield {
                    "type": "debug",
                    "data": {
                        "intent": intent.value,
                        "rewritten_query": rewritten_query,
                        "sources": debug_sources,
                        "context_length": sum(len(d.get("content", "")) for d in valid_docs),
                        "note": f"Threshold({SIMILARITY_THRESHOLD}): {len(valid_docs)}/{len(docs)} 건 사용"
                    }
                }
            
            # Context 생성
            if not valid_docs:
                context = "관련 상담 내역 없음. 일반 지식 기반 응답."
            else:
                context = format_sources(valid_docs)
            
            # Answer 스트리밍
            full_answer = ""
            for chunk in self.answer_chain.stream({
                "context": context,
                "history": history_text,
                "query": rewritten_query
            }):
                filtered = filter_special_tokens(chunk)
                if not filtered:
                    continue
                full_answer += filtered
                if debug:
                    yield {"type": "content", "data": filtered}
                else:
                    yield filtered
            
            # 후처리
            clean_answer = full_answer
            if "[EXPERT_REFERRAL_NEEDED]" in full_answer:
                clean_answer = full_answer.replace("[EXPERT_REFERRAL_NEEDED]", "").strip()
                self._handle_expert_referral(session_id, clean_answer)
            
            self.db.add_chat_message(session_id, "assistant", clean_answer)
            print("[LangGraphRAG] Stream End")
            
        except Exception as e:
            print(f"[LangGraphRAG] Stream Error: {e}")
            err_msg = "죄송합니다. 처리 중 오류가 발생했습니다."
            if debug:
                yield {"type": "content", "data": err_msg}
            else:
                yield err_msg
    
    async def stream_async(self, user_id: int, session_id: int, query: str, debug: bool = False):
        """비동기 스트리밍 (기존 RAGChain.stream_async 호환)"""
        import asyncio
        
        def sync_stream():
            return list(self.stream(user_id, session_id, query, debug))
        
        chunks = await asyncio.to_thread(sync_stream)
        for chunk in chunks:
            yield chunk
    
    def run_with_debug(self, query: str, history: List[Dict[str, str]] = []) -> Dict[str, Any]:
        """디버그 정보 포함 실행"""
        history_text = format_history(history)
        
        initial_state: RAGState = {
            "query": query,
            "user_id": 0,
            "session_id": 0,
            "history_text": history_text,
            "intent": "",
            "needs_rag": True,
            "direct_response": "",
            "rewritten_query": "",
            "source_docs": [],
            "context": "",
            "is_low_similarity": False,
            "answer": ""
        }
        
        try:
            result = self.graph.invoke(initial_state)
            return {
                "input_query": query,
                "rewritten_query": result.get("rewritten_query", ""),
                "source_docs": result.get("source_docs", []),
                "context": result.get("context", ""),
                "answer": result.get("answer", "")
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ---------------------------------------------------------
    # Private Helpers
    # ---------------------------------------------------------
    
    def _handle_expert_referral(self, session_id: int, answer: str):
        """전문가 연결 DB 기록"""
        try:
            self.db.create_expert_referral(
                session_id=session_id,
                severity_level="high",
                recommended_action="전문 상담사 연결 권장"
            )
        except Exception as e:
            print(f"[LangGraphRAG] 전문가 연결 로깅 실패: {e}")
    
    def _generate_session_summary(self, session_id: int) -> str:
        """세션 요약 생성"""
        history = self.db.get_chat_history(session_id)
        if not history:
            return "진행된 상담 내역이 없어 요약할 내용이 없습니다."
        
        conversation = ""
        for msg in history:
            if msg.role == "system":
                continue
            role = "상담사" if msg.role == "assistant" else "내담자"
            conversation += f"{role}: {msg.content}\n"
        
        prompt = f"""
[역할] 심리 상담 요약 AI
[대화]
{conversation}

[규칙]
1. 상담사의 구체적 조언/기법만 추출
2. [오늘의 심리 처방] 형식으로 요약
3. 마지막에 따뜻한 격려
"""
        response = self.model.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)


# -------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------

if __name__ == "__main__":
    print("=== LangGraphRAG Test ===")
    
    db = DatabaseManager(echo=False)
    rag = LangGraphRAG(db_manager=db)
    
    # 테스트 사용자/세션
    try:
        user = db.create_user("test_langgraph_user")
    except:
        user = db.get_user_by_username("test_langgraph_user")
    
    session = db.create_chat_session(user.id)
    
    # 테스트 쿼리
    test_queries = ["안녕", "요즘 힘들어", "그만 둘까 고민이야"]
    
    for q in test_queries:
        print(f"\n[Test] Query: {q}")
        answer = rag.run(user.id, session.id, q)
        print(f"[Bot] {answer[:100]}...")
