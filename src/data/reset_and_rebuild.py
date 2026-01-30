"""
FileName    : reset_and_rebuild.py
Auth        : 박수빈
Date        : 2026-01-28
Description : 데이터 파이프라인 리셋 및 재처리 통합 스크립트
Issue/Note  : 기존 데이터 삭제 → 전처리 → 임베딩 자동화
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

import os
import sys
import shutil
import argparse
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.db_config import DatabaseConfig


# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

# 삭제 대상 경로
PATHS_TO_DELETE = [
    DatabaseConfig.PROCESSED_DATA_DIR,
    DatabaseConfig.CHROMA_DB_DIR,
    DatabaseConfig.SQLITE_DB_PATH,
]

# 전처리 기본 경로
DEFAULT_TXT_ROOT = PROJECT_ROOT / "data" / "raw" / "16.심리상담 데이터" / "3.개방데이터" / "1.데이터" / "Training" / "01.원천데이터"
DEFAULT_JSON_ROOT = PROJECT_ROOT / "data" / "raw" / "16.심리상담 데이터" / "3.개방데이터" / "1.데이터" / "Training" / "02.라벨링데이터"
DEFAULT_OUT_DIR = DatabaseConfig.PROCESSED_DATA_DIR


# -------------------------------------------------------------
# Step 1: Clean existing data
# -------------------------------------------------------------

def clean_existing_data(dry_run: bool = False) -> None:
    """
    기존 처리 데이터 삭제
    
    Args:
        dry_run: True면 삭제하지 않고 대상만 출력
    """
    print("\n" + "=" * 60)
    print("STEP 1: 기존 데이터 정리")
    print("=" * 60)
    
    for path in PATHS_TO_DELETE:
        path = Path(path)
        if path.exists():
            if dry_run:
                print(f"   [DRY-RUN] 삭제 예정: {path}")
            else:
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"   ✅ 삭제됨 (디렉토리): {path}")
                else:
                    path.unlink()
                    print(f"   ✅ 삭제됨 (파일): {path}")
        else:
            print(f"   ⏭️  존재하지 않음: {path}")


# -------------------------------------------------------------
# Step 2: Run preprocessing
# -------------------------------------------------------------

def run_preprocessing(txt_root: Path, json_root: Path, out_dir: Path, window: int = 1) -> bool:
    """
    전처리 파이프라인 실행 (subprocess 사용)
    
    Args:
        txt_root: 원천 텍스트 데이터 경로
        json_root: 라벨링 JSON 데이터 경로
        out_dir: 출력 디렉토리
        window: 컨텍스트 윈도우 크기
    
    Returns:
        성공 여부
    """
    import subprocess
    
    print("\n" + "=" * 60)
    print("STEP 2: 데이터 전처리")
    print("=" * 60)
    
    try:
        preprocess_script = PROJECT_ROOT / "src" / "data" / "preprocess_data.py"
        cmd = [
            sys.executable,
            str(preprocess_script),
            "--txt_root", str(txt_root),
            "--json_root", str(json_root),
            "--out_dir", str(out_dir),
            "--window", str(window)
        ]
        
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"   ❌ 전처리 실패: {result.stderr}")
            return False
        
        print(result.stdout)
        print("   ✅ 전처리 완료")
        return True
    except Exception as e:
        print(f"   ❌ 전처리 실패: {e}")
        return False



# -------------------------------------------------------------
# Step 3: Run embedding
# -------------------------------------------------------------

def run_embedding(input_path: Path, batch_size: int = 100, limit: int = None) -> bool:
    """
    벡터DB 임베딩 실행
    
    Args:
        input_path: 입력 JSONL 파일 경로
        batch_size: 배치 크기
        limit: 테스트용 최대 문서 수
    
    Returns:
        성공 여부
    """
    print("\n" + "=" * 60)
    print("STEP 3: 벡터DB 임베딩")
    print("=" * 60)
    
    if not input_path.exists():
        print(f"   ❌ 입력 파일 없음: {input_path}")
        return False
    
    try:
        from src.data.embed_to_vectordb import embed_documents
        stats = embed_documents(
            input_path=input_path,
            batch_size=batch_size,
            limit=limit
        )
        print(f"   ✅ 임베딩 완료: {stats}")
        return True
    except Exception as e:
        print(f"   ❌ 임베딩 실패: {e}")
        return False


# -------------------------------------------------------------
# Step 4: Print summary
# -------------------------------------------------------------

def print_summary(out_dir: Path) -> None:
    """
    처리 결과 요약 출력
    
    Args:
        out_dir: 출력 디렉토리
    """
    print("\n" + "=" * 60)
    print("STEP 4: 결과 요약")
    print("=" * 60)
    
    summary_path = out_dir / "docs_summary.txt"
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"))
    else:
        print("   ⚠️  요약 파일 없음")
    
    # VectorDB 문서 수 확인
    try:
        from src.database import VectorStore
        vs = VectorStore()
        count = vs.get_document_count()
        print(f"\n   📊 VectorDB 문서 수: {count:,}")
    except Exception as e:
        print(f"   ⚠️  VectorDB 조회 실패: {e}")


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main(
    txt_root: str = None,
    json_root: str = None,
    out_dir: str = None,
    window: int = 1,
    batch_size: int = 100,
    limit: int = None,
    dry_run: bool = False,
    skip_clean: bool = False,
    skip_preprocess: bool = False,
    skip_embed: bool = False
) -> None:
    """
    데이터 파이프라인 전체 실행
    """
    txt_root = Path(txt_root) if txt_root else DEFAULT_TXT_ROOT
    json_root = Path(json_root) if json_root else DEFAULT_JSON_ROOT
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
    
    print("\n" + "=" * 60)
    print("🔄 데이터 파이프라인 리셋 및 재처리")
    print("=" * 60)
    print(f"   TXT 경로: {txt_root}")
    print(f"   JSON 경로: {json_root}")
    print(f"   출력 경로: {out_dir}")
    print(f"   윈도우 크기: {window}")
    print(f"   배치 크기: {batch_size}")
    if limit:
        print(f"   제한: {limit}개")
    if dry_run:
        print("   ⚠️  DRY-RUN 모드")
    
    # Step 1: Clean
    if not skip_clean:
        clean_existing_data(dry_run=dry_run)
    
    if dry_run:
        print("\n   [DRY-RUN] 실제 처리는 수행하지 않음")
        return
    
    # Step 2: Preprocess
    if not skip_preprocess:
        success = run_preprocessing(txt_root, json_root, out_dir, window)
        if not success:
            print("\n❌ 전처리 실패로 중단")
            return
    
    # Step 3: Embed
    if not skip_embed:
        input_path = out_dir / "docs_for_vectordb.jsonl"
        success = run_embedding(input_path, batch_size, limit)
        if not success:
            print("\n❌ 임베딩 실패로 중단")
            return
    
    # Step 4: Summary
    print_summary(out_dir)
    
    print("\n" + "=" * 60)
    print("✅ 파이프라인 완료!")
    print("=" * 60)


# -------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="데이터 파이프라인 리셋 및 재처리")
    parser.add_argument("--txt_root", type=str, default=None, help="원천 데이터 경로")
    parser.add_argument("--json_root", type=str, default=None, help="라벨링 데이터 경로")
    parser.add_argument("--out_dir", type=str, default=None, help="출력 디렉토리")
    parser.add_argument("--window", type=int, default=1, help="컨텍스트 윈도우 크기")
    parser.add_argument("--batch_size", type=int, default=100, help="임베딩 배치 크기")
    parser.add_argument("--limit", type=int, default=None, help="테스트용 최대 문서 수")
    parser.add_argument("--dry-run", action="store_true", help="실제 실행 없이 확인만")
    parser.add_argument("--skip-clean", action="store_true", help="기존 데이터 삭제 건너뛰기")
    parser.add_argument("--skip-preprocess", action="store_true", help="전처리 건너뛰기")
    parser.add_argument("--skip-embed", action="store_true", help="임베딩 건너뛰기")
    
    args = parser.parse_args()
    
    main(
        txt_root=args.txt_root,
        json_root=args.json_root,
        out_dir=args.out_dir,
        window=args.window,
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
        skip_clean=args.skip_clean,
        skip_preprocess=args.skip_preprocess,
        skip_embed=args.skip_embed
    )
