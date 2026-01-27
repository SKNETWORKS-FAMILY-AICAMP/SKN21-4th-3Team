"""
FileName    : pdf_exporter.py
Auth        : 우재현
Date        : 2026-01-08
Description : 사용자와 상담자의 대화기록을 pdf화 시켜 받을 수 있게하는 기능
Issue/Note  : 
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Root 경로 설정
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database.db_manager import DatabaseManager
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm

# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------

# 폰트 경로 (Windows 기본 폰트 '맑은 고딕' 사용)
FONT_NAME = "Malgun"
FONT_PATH = "C:/Windows/Fonts/malgun.ttf" 
HISTORY_DIR = str(Path(__file__).parent.parent.parent / "history")

# -------------------------------------------------------------
# Class
# -------------------------------------------------------------

class PDFExporter:
    """
    채팅 히스토리를 PDF로 변환하는 클래스
    """
    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager if db_manager else DatabaseManager()
        self._register_font()

    def _register_font(self):
        """
        한글 폰트 등록
        """
        if os.path.exists(FONT_PATH):
            try:
                pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
                self.font_available = True
            except Exception as e:
                print(f"[Warning] 폰트 등록 실패: {e}")
                self.font_available = False
        else:
            print(f"[Warning] 폰트 파일이 없습니다: {FONT_PATH}")
            self.font_available = False

    def export_session(self, session_id: int, output_filename: str = None) -> str:
        """
        특정 세션의 대화 내용을 PDF로 저장
        
        Args:
            session_id: 세션 ID
            output_filename: 저장할 파일명 (None이면 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        # 1. 대화 내용 조회
        history = self.db.get_chat_history(session_id)
        if not history:
            print(f"[Info] 세션 {session_id}의 대화 내용이 없습니다.")
            return None

        # 2. 파일명 설정 (자동 생성 시)
        if output_filename is None:
            today_str = datetime.now().strftime("%Y-%m-%d")
            output_filename = f"{today_str}_{session_id}.pdf"
        
        # 3. 저장 경로 설정 (history 폴더 사용)
        # 만약 output_filename이 절대 경로가 아니라면 history 폴더에 저장
        if not os.path.isabs(output_filename):
            # history 폴더 확인 및 생성 (없을 경우 대비)
            if not os.path.exists(HISTORY_DIR):
                try:
                    os.makedirs(HISTORY_DIR)
                except OSError:
                    pass # 권한 문제 등 무시하고 현재 경로 사용 가능성 열어둠
            
            output_filename = os.path.join(HISTORY_DIR, output_filename)

        # 3. PDF 생성
        try:
            self._create_pdf_canvas(output_filename, history)
            print(f"[Success] PDF 저장 완료: {output_filename}")
            return output_filename
        except Exception as e:
            print(f"[Error] PDF 생성 실패: {e}")
            return None

    def _create_pdf_canvas(self, filename, history):
        """
        ReportLab Canvas를 사용하여 PDF 드로잉
        """
        c = canvas.Canvas(filename, pagesize=A4)
        width, height = A4
        
        # 폰트 설정
        if self.font_available:
            c.setFont(FONT_NAME, 10)
        
        # 타이틀
        c.setFont(FONT_NAME, 14) if self.font_available else None
        c.drawString(50, height - 50, f"상담 기록 (Session {history[0].session_id})")
        
        # 라인
        c.line(50, height - 60, width - 50, height - 60)
        
        y = height - 100
        line_height = 16
        margin = 50
        max_width = width - (margin * 2)

        if self.font_available:
            c.setFont(FONT_NAME, 10)

        for msg in history:
            role = "나" if msg.role == "user" else "상담사"
            content = msg.content
            
            # 롤 표시
            c.setFont(FONT_NAME, 10) if self.font_available else None
            # 색상 다르게 (User: 검정, Bot: 파랑)
            if msg.role == "user":
                c.setFillColorRGB(0, 0, 0)
            else:
                c.setFillColorRGB(0, 0, 0.5)

            c.drawString(margin, y, f"[{role}]")
            y -= line_height

            # 내용 표시 (간단한 줄바꿈 처리)
            c.setFillColorRGB(0.2, 0.2, 0.2)
            
            # 긴 텍스트 줄바꿈 (단순 구현, wrap 로직 필요하면 Paragraph 사용 권장)
            # 여기서는 간단히 50자마다 줄바꿈
            chars_per_line = 60
            lines = [content[i:i+chars_per_line] for i in range(0, len(content), chars_per_line)]
            
            for line in lines:
                if y < 50: # 페이지 넘기기
                    c.showPage()
                    y = height - 50
                    if self.font_available:
                        c.setFont(FONT_NAME, 10)
                
                c.drawString(margin + 20, y, line)
                y -= line_height
            
            y -= 10 # 메시지 간 간격
            
            if y < 50:
                c.showPage()
                y = height - 50
                if self.font_available:
                    c.setFont(FONT_NAME, 10)

        c.save()

# -------------------------------------------------------------
# Main Functions
# -------------------------------------------------------------

def main():
    """
    Test Main Function
    """
    print("=== PDF Exporter Test ===")
    
    # DB 연결 (테스트용)
    db = DatabaseManager(echo=False)
    exporter = PDFExporter(db)
    
    # 최근 세션 가져오기
    user = db.get_user_by_username("test_user_001") # 기존 테스트 유저 활용
    if not user:
        # 유저가 없으면 생성 시도 (DB 상태에 따라 다름)
        try:
             user = db.create_user("test_pdf_user")
             session = db.create_chat_session(user.id)
             db.add_chat_message(session.id, "user", "안녕하세요. 테스트 메시지입니다.")
             db.add_chat_message(session.id, "assistant", "반갑습니다. PDF 저장을 테스트합니다.")
             target_session_id = session.id
        except:
             print("[Error] 테스트 유저/세션 생성 불가")
             return
    else:
        # 최근 세션 조회
        sessions = db.get_user_recent_sessions(user.id, limit=1)
        if sessions:
            target_session_id = sessions[0]["id"]
        else:
             print("[Error] 테스트할 세션이 없습니다.")
             return

    print(f"Exporting Session ID: {target_session_id}")
    
    # Export 실행
    # output_path = None 으로 설정하여 자동 생성 및 history 폴더 저장 테스트
    result = exporter.export_session(target_session_id)
    
    if result:
        print(f"File saved at: {result}")
    else:
        print("Export failed.")

if __name__ == "__main__":
    main()
