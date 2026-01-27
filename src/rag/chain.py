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
        """
        # (기존 로직 유지하되 run_with_debug 호출로 대체 가능하나, 안전하게 기존 구조 유지)
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
            # 3. 체인 실행
            result = self.rag_pipeline.invoke({
                "query": query,
                "history_text": history_text
            })
            
            answer = result["answer"].strip()
            # ... (나머지 후처리 로직)
            
            # 4. 전문가 연결 감지 (후처리)
            if "[EXPERT_REFERRAL_NEEDED]" in answer:
                answer = answer.replace("[EXPERT_REFERRAL_NEEDED]", "").strip()
                self._handle_expert_referral(session_id, answer)
                if "상담" not in answer:
                    answer += "\n"
                    # answer += "\n\n(전문가와의 상담이 필요해 보여 전문 상담 센터 정보를 준비하고 있습니다.)"
            
            # 5. Assistant 메시지 저장
            self.db.add_chat_message(session_id, "assistant", answer)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"[System] Response Time: {elapsed_time:.2f}s")
            
            print(f"[Flow End] 답변 등 생성 완료")
            return answer
            
        except Exception as e:
            print(f"[Error] RAG 파이프라인 실패: {e}")
            return "죄송합니다. 처리 중 오류가 발생했습니다."

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
        user = test_db.get_user_by_username("test_lcel_user_01")
        
    session = test_db.create_chat_session(user.id)
    
    # 2. Run Flow
    q1 = "사는게 재미가 없어"
    ans1 = rag_chain.run(user.id, session.id, q1)
    print(f"\n[Bot]: {ans1}\n")

