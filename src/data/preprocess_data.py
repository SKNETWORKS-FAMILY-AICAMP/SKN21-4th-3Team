"""
FileName    : preprocess_data.py
Auth        : 장이선
Date        : 2026-01-06
Description : 원본 txt/json 데이터 전처리
Issue/Note  : 비고
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

import os
import glob
import json
import re
import argparse
from collections import defaultdict
import pandas as pd
from pathlib import Path


# -------------------------------------------------------------
# STEP 1. File reading utilities
# -------------------------------------------------------------

def read_text_file(path:str)->str:
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        return ""

    if not raw:
        return ""

    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            pass

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass

    return raw.decode("cp949", errors="replace")


def read_json_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    
# -------------------------------------------------------------
# STEP 2. Pattern definitions & extractors
# -------------------------------------------------------------

id_pattern = re.compile(r'(?<![A-Z0-9])([DAXN]\d{3})(?!\d)')
session_pattern_kr = re.compile(r'(\d+)\s*회기')
session_pattern_en = re.compile(r'session[_\s-]?(\d+)', re.IGNORECASE)
speaker_pattern = re.compile(r'^(상담사|내담자|상담자|상담가|치료자|고객|사용자|T|C)\s*[:：]\s*(.*)$', re.IGNORECASE)
fallback_colon_pattern = re.compile(r'^(.{1,30}?)\s*[:：]\s*(.*)$')

def extract_source_id(path):
    m = id_pattern.search(path)
    return m.group(1) if m else None


def extract_session_no(path):
    m = session_pattern_kr.search(path)
    if m:
        return int(m.group(1))
    m = session_pattern_en.search(path)
    if m:
        return int(m.group(1))
    return None

def extract_category(path):
    p = path.lower()
    if ("우울" in path) or ("depression" in p):
        return "DEPRESSION"
    if ("불안" in path) or ("anxiety" in p):
        return "ANXIETY"
    if ("중독" in path) or ("addiction" in p):
        return "ADDICTION"
    if ("일반" in path) or ("normal" in p):
        return "NORMAL"
    return None

# -------------------------------------------------------------
# STEP 2-1. Dataset construction helpers (normalize + window chunk)
# -------------------------------------------------------------

tag_rules = [
    (re.compile(r"@COUNSELOR\b", re.IGNORECASE), "상담사"),
    (re.compile(r"@NAME\b", re.IGNORECASE), "내담자"),
    (re.compile(r"@TIME\b", re.IGNORECASE), "[TIME]"),
    (re.compile(r"@DATE\b", re.IGNORECASE), "[DATE]"),
    (re.compile(r"@YEAR\b", re.IGNORECASE), "[DATE]"),
    (re.compile(r"@AGE\b", re.IGNORECASE), "[AGE]"),
    (re.compile(r"@PLACE\b", re.IGNORECASE), "[PLACE]"),
    (re.compile(r"@SCHOOL\b", re.IGNORECASE), "[PLACE]"),
    (re.compile(r"@DEPARTMENT\b", re.IGNORECASE), "[PLACE]"),
    (re.compile(r"@COMPANY\b", re.IGNORECASE), "[PLACE]"),
    (re.compile(r"@TITLE\b", re.IGNORECASE), "[TITLE]"),
    (re.compile(r"@ETC\b", re.IGNORECASE), "[ETC]"),
]

def normalize_text(text: str) -> str:
    t = text
    for pat, rep in tag_rules:
        t = pat.sub(rep, t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def build_window_text(chunks, i: int, window: int = 1) -> str:
    start = max(0, i - window)
    end = min(len(chunks) - 1, i + window)

    parts = []
    for j in range(start, end + 1):
        spk = chunks[j].get("speaker", "UNKNOWN")
        utt = chunks[j].get("utterance", "")
        parts.append(f"{spk}: {normalize_text(utt)}")

    return "\n".join(parts).strip()

# -------------------------------------------------------------
# STEP 3. Text chunking
# -------------------------------------------------------------

def chunk_dialogue(text:str):
    lines = text.splitlines()
    chunks = []
    cur_speaker = None
    cur_utt = []

    def flush():
        nonlocal cur_speaker, cur_utt
        if cur_speaker is not None:
            utt = "\n".join(cur_utt).strip()
            if utt:
                chunks.append({"speaker": cur_speaker, "utterance": utt})
        cur_speaker = None
        cur_utt = []

    def norm(spk: str) -> str:
        if spk is None:
            return "UNKNOWN"
        s = spk.strip()
        if s.isdigit():
            return "UNKNOWN"
        su = s.upper()
        if su == "T":
            return "상담사"
        if su == "C":
            return "내담자"
        if s in {"상담자", "상담가", "치료자"}:
            return "상담사"
        if s in {"고객", "사용자"}:
            return "내담자"
        return s

    for line in lines:
        s = line.strip()
        if not s:
            continue

        m = speaker_pattern.match(s)
        if m:
            flush()
            cur_speaker = norm(m.group(1))
            cur_utt = [m.group(2)]
            continue

        fm = fallback_colon_pattern.match(s)
        if fm:
            flush()
            cur_speaker = norm(fm.group(1))
            cur_utt = [fm.group(2)]
            continue

        # ✅ 어떤 패턴에도 안 걸려도 "버리지 말고" 기존 턴에 이어붙이기
        if cur_speaker is None:
            cur_speaker = "UNKNOWN"
            cur_utt = [s]
        else:
            cur_utt.append(s)

    flush()
    return chunks


# -------------------------------------------------------------
# STEP 4. Label parsing & validation
# -------------------------------------------------------------

def find_first_list_in_dict(d):
    for v in d.values():
        if isinstance(v, list):
            return v
    return []


def parse_labels_from_json(j):
    if isinstance(j, list):
        return j
    if isinstance(j, dict):
        return find_first_list_in_dict(j)
    return []


def validate_alignment(chunks, labels):
    if not chunks:
        return False, "no_chunks"
    if not labels:
        return False, "no_labels"
    if len(chunks) != len(labels):
        return False, f"len_mismatch ({len(chunks)} vs {len(labels)})"
    return True, "ok"

# -------------------------------------------------------------
# STEP 5. Session payload builder
# -------------------------------------------------------------

def build_payload(row):
    text = read_text_file(row["txt_path"])
    chunks = chunk_dialogue(text)
    text = re.sub(r'(?<!^)\s*(상담사|내담자|상담자|상담가|치료자|고객|사용자|T|C)\s*[:：]\s*',
              r'\n\1: ', text, flags=re.IGNORECASE)


    j, jerr = read_json_safe(row["json_path"])
    if j is None:
        return {
            **row,
            "chunks": chunks,
            "labels": [],
            "align_ok": False,
            "align_msg": f"json_load_fail | {jerr}",
        }

    labels = parse_labels_from_json(j)
    ok, msg = validate_alignment(chunks, labels)

    return {
        **row,
        "chunks": chunks,
        "labels": labels,
        "align_ok": ok,
        "align_msg": msg,
    }

# -------------------------------------------------------------
# STEP 6. Main processing pipeline
# -------------------------------------------------------------

def main(txt_root, json_root, out_dir, window: int = 1):
    txt_files = glob.glob(os.path.join(txt_root, "**", "*.txt"), recursive=True)
    json_files = glob.glob(os.path.join(json_root, "**", "*.json"), recursive=True)

    txt_map = defaultdict(list)
    json_map = defaultdict(list)

    for p in txt_files:
        sid = extract_source_id(p)
        sess = extract_session_no(p)
        if sid and sess is not None:
            txt_map[(sid, sess)].append(p)

    for p in json_files:
        sid = extract_source_id(p)
        sess = extract_session_no(p)
        if sid and sess is not None:
            json_map[(sid, sess)].append(p)

    keys = sorted(set(txt_map.keys()) | set(json_map.keys()))
    rows = []

    for sid, sess in keys:
        tlist = txt_map.get((sid, sess), [])
        jlist = json_map.get((sid, sess), [])
        tpath = tlist[0] if tlist else None
        jpath = jlist[0] if jlist else None

        if tlist and jlist:
            status = "MATCH"
        elif tlist:
            status = "MISSING_JSON"
        elif jlist:
            status = "MISSING_TXT"
        else:
            status = "UNKNOWN"

        rows.append({
            "session_id": f"{sid}__S{sess:03d}",
            "source_id": sid,
            "session_no": sess,
            "category": extract_category(tpath or jpath or ""),
            "status": status,
            "txt_path": tpath,
            "json_path": jpath,
        })

    df = pd.DataFrame(rows)
    df_match = df[df["status"] == "MATCH"].copy()

    payloads = [build_payload(r) for r in df_match.to_dict(orient="records")]

    ok_payloads = [x for x in payloads if x["align_ok"]]
    fail_payloads = [x for x in payloads if not x["align_ok"]]

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "sessions_OK.jsonl"), "w", encoding="utf-8") as f:
        for x in ok_payloads:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    pd.DataFrame([{
        "session_id": x["session_id"],
        "reason": x["align_msg"],
        "txt_path": x["txt_path"],
        "json_path": x["json_path"],
    } for x in fail_payloads]).to_csv(
        os.path.join(out_dir, "sessions_FAIL_report.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    # -------------------------------------------------------------
    # STEP 6-2. Build VectorDB documents (docs_for_vectordb.jsonl)
    # -------------------------------------------------------------

    docs = []

    # ✅ 벡터DB 문서는 align_ok와 분리:
    #    - chunks가 있으면 문서를 만든다
    #    - labels가 있으면 metadata에 category로 넣고, 없으면 None
    docs_payloads = [x for x in payloads if x.get("chunks")]

    for s in docs_payloads:
        session_id = s["session_id"]
        source_id = s["source_id"]
        category_default = s.get("category", "")
        chunks = s["chunks"]
        labels = s.get("labels", [])

        for i in range(len(chunks)):
            text = build_window_text(chunks, i, window=window)

            # 라벨이 있으면 사용, 없으면 None
            category = labels[i] if i < len(labels) else None
            if category is None:
                category = None  # 명시적으로 None 유지 (원하면 "UNLABELED"로 바꿔도 됨)

            docs.append({
                "text": text,
                "metadata": {
                    "session_id": session_id,
                    "source_id": source_id,
                    "default_category": category_default,
                    "category": category,
                    "turn_index": i,
                    "align_ok": bool(s.get("align_ok", False)),
                }
            })


    docs_path = os.path.join(out_dir, "docs_for_vectordb.jsonl")
    with open(docs_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # -------------------------------------------------------------
    # STEP 6-3. Save dataset summary (docs_summary.txt)
    # -------------------------------------------------------------

    lengths = [len(d["text"]) for d in docs]
    avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0

    summary_lines = [
        f"정렬 OK 세션 수: {len(ok_payloads)}",
        f"문서 생성 세션 수(chunks 기준): {len(docs_payloads)}",
        f"총 문서 수: {len(docs)}",
        f"Chunk 기준: window(i-{window}~i+{window}), 중심 turn 기준 1문서",
        f"평균 문서 길이(char): {avg_len:.1f}",
        f"저장 파일: {docs_path}",
    ]

    with open(os.path.join(out_dir, "docs_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

# -------------------------------------------------------------
# STEP 7. CLI entrypoint
# -------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_root", default="data/raw/16.심리상담 데이터/3.개방데이터/1.데이터/Training/01.원천데이터")
    parser.add_argument("--json_root", default="data/raw/16.심리상담 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--window", type=int, default=1)

    args = parser.parse_args()
    main(args.txt_root, args.json_root, args.out_dir, window=args.window)
