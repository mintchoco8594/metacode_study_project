# app.py
import json
import re
from pathlib import Path

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer

import utils.type_calc as type_calc
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()
STORE_DIR = Path("src/rag_store")
INDEX_PATH = STORE_DIR / "pokemon.index"
DOCS_PATH = STORE_DIR / "documents.jsonl"
TYPE_CHART_PATH = Path("src/data/type_chart.json")

# ---------------- LangChain LLM (해설용) ----------------
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
    )

# ---------------- Embedding model (FAISS 검색용) ----------------
@st.cache_resource
def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_faiss_index():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")
    return faiss.read_index(str(INDEX_PATH))

@st.cache_data
def load_docs():
    if not DOCS_PATH.exists():
        raise FileNotFoundError(f"documents.jsonl not found: {DOCS_PATH}")
    docs = []
    for line in DOCS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            docs.append(json.loads(line))
    return docs

@st.cache_data
def build_name_index():
    docs = load_docs()

    ko_names = set()
    en_names = set()

    for d in docs:
        m = d.get("meta", {})
        ko = (m.get("korean_name") or "").strip()
        en = (m.get("english_name") or "").strip().lower()
        if ko:
            ko_names.add(ko)
        if en:
            en_names.add(en)

    # 긴 이름이 먼저 매칭되게 길이순 정렬(부분 포함 이슈 줄임)
    ko_list = sorted(ko_names, key=len, reverse=True)
    en_list = sorted(en_names, key=len, reverse=True)
    return ko_list, en_list

@st.cache_data
def load_chart():
    if not TYPE_CHART_PATH.exists():
        raise FileNotFoundError(f"type_chart.json not found: {TYPE_CHART_PATH}")
    return type_calc.load_type_chart(TYPE_CHART_PATH)

def search_docs(query: str, k: int = 8):
    embedder = get_embedder()          # ✅ 임베더
    index = load_faiss_index()
    docs = load_docs()

    q = embedder.encode([query], normalize_embeddings=True).astype("float32")  # ✅ 여기!
    D, I = index.search(q, k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(docs):
            continue
        results.append((float(score), docs[idx]))
    return results

def normalize_name(s: str) -> str:
    return (s or "").strip()

def clean_name_token(name: str, valid_types: set[str]) -> str:
    s = (name or "").strip()

    # 한글 타입 단어 제거 (노말/불꽃/전기 등)
    for ko in type_calc.TYPE_ALIAS_KO.keys():
        s = s.replace(ko, " ")

    # 영문 타입 단어 제거 (fire, water 등)
    for ty in valid_types:
        s = re.sub(rf"\b{re.escape(ty)}\b", " ", s, flags=re.IGNORECASE)

    # 자잘한 조사/표현 제거 (원하면 더 추가)
    s = re.sub(r"(상대로|에게|한테|사용|기술|공격|쓰면|써|로)\b", " ", s)

    # 공백 정리
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_pokemon_names_from_text(text: str, max_n: int = 2) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []

    ko_list, en_list = build_name_index()
    found = []
    used = set()

    # 1) 한글 이름: 문장에 포함되면 채택
    for name in ko_list:
        if name in raw and name not in used:
            found.append(name)
            used.add(name)
            if len(found) >= max_n:
                return found

    # 2) 영문 이름: 단어 경계 기준(대충)으로 채택
    low = raw.lower()
    for en in en_list:
        # 너무 짧은 영어는 오탐 많아서 제외(원하면 3~4로)
        if len(en) < 4:
            continue
        if re.search(rf"\b{re.escape(en)}\b", low) and en not in used:
            found.append(en)
            used.add(en)
            if len(found) >= max_n:
                return found

    return found

def safe_types(meta: dict) -> list[str]:
    if isinstance(meta.get("types"), list) and meta["types"]:
        return [t for t in meta["types"] if t]
    return []

def find_pokemon_by_name(name: str):
    name = normalize_name(name)
    if not name:
        return None

    docs = load_docs()
    low = name.lower()

    for d in docs:
        m = d.get("meta", {})
        if (m.get("korean_name") or "").strip() == name:
            return 1.0, d
        if (m.get("english_name") or "").strip().lower() == low:
            return 1.0, d

    hits = search_docs(name, k=5)
    if not hits:
        return None
    score, doc = hits[0]
    if score < 0.35:
        return None
    return score, doc


def parse_battle_text(text: str, valid_types: set[str]) -> dict:
    raw = (text or "").strip()
    if not raw:
        return {"a": "", "b": "", "move_type": None}
    move_type = type_calc.guess_move_type_from_text(raw, valid_types)

    names = extract_pokemon_names_from_text(raw, max_n=2)
    if len(names) >= 2:
        return {"a": names[0], "b": names[1], "move_type": move_type} 
    # 1) A vs B / A 대 B
    m = re.search(r"(.+?)\s*(?:vs|VS|대)\s*(.+)", raw)
    if m:
        a = clean_name_token(m.group(1).strip(), valid_types)
        b = clean_name_token(m.group(2).strip(), valid_types)
        move_type = type_calc.guess_move_type_from_text(raw, valid_types)
        return {"a": a, "b": b, "move_type": move_type}

    # 2) "A가 B를/을/상대로/에게/한테"
    m = re.search(r"(.+?)(?:이|가)\s*(.+?)(?:를|을|상대로|에게|한테)", raw)
    if m:
        a = clean_name_token(m.group(1).strip(), valid_types)
        b = clean_name_token(m.group(2).strip(), valid_types)
        move_type = type_calc.guess_move_type_from_text(raw, valid_types)
        return {"a": a, "b": b, "move_type": move_type}

    # 3) fallback: 첫 두 토큰
    parts = raw.split()
    a = clean_name_token(parts[0], valid_types) if len(parts) >= 1 else ""
    b = clean_name_token(parts[1], valid_types) if len(parts) >= 2 else ""
    move_type = type_calc.guess_move_type_from_text(raw, valid_types)
    return {"a": a, "b": b, "move_type": move_type}



def render_result(a_doc: dict, b_doc: dict, move_type: str | None, mult: float | None, label: str | None):
    st.divider()
    st.subheader("포켓몬 상성 분석")
    st.write(f"A: {a_doc['meta'].get('korean_name') or a_doc['meta'].get('english_name') or ''}")
    st.write(f"B: {b_doc['meta'].get('korean_name') or b_doc['meta'].get('english_name') or ''}")
    st.write(f"공격 타입: {move_type or 'normal'}")

    if move_type and mult is not None:
        st.write(f"상성 배율: {mult:.2f}x")
        st.write(f"상성 레이블: {label}")
    else:
        st.write("공격 타입을 찾을 수 없어요.")
    
def build_rag_context(a_doc: dict, b_doc: dict) -> str:
    a_meta = a_doc.get("meta", {})
    b_meta = b_doc.get("meta", {})

    a_name = a_meta.get("korean_name") or a_meta.get("english_name") or ""
    b_name = b_meta.get("korean_name") or b_meta.get("english_name") or ""

    a_types = a_meta.get("types", [])
    b_types = b_meta.get("types", [])

    a_stats = a_meta.get("stats", {})
    b_stats = b_meta.get("stats", {})

    # text는 너무 길면 토큰 터지니까 자르자
    a_text = (a_doc.get("text", "") or "")[:800]
    b_text = (b_doc.get("text", "") or "")[:800]

    return f"""
[포켓몬 A 근거]
이름: {a_name}
타입: {a_types}
스탯: {a_stats}
설명: {a_text}

[포켓몬 B 근거]
이름: {b_name}
타입: {b_types}
스탯: {b_stats}
설명: {b_text}
""".strip()


# ---------------- UI ----------------
st.set_page_config(page_title="Pokemon Battle Chat (RAG+Rules)", layout="wide")
st.title("포켓몬 배틀 시뮬레이션 챗봇")

chart = load_chart()
valid_types = set(chart.keys())

if "chat" not in st.session_state:
    st.session_state.chat = []

prompt = st.chat_input("예) 리자몽이 이상해꽃 상대로 불꽃 기술 쓰면 어때? / 피카츄 vs 꼬부기 전기")

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})

    parsed = parse_battle_text(prompt, valid_types)
    a_name = parsed["a"]
    b_name = parsed["b"]
    move_type = parsed["move_type"] or "normal"

    a_hit = find_pokemon_by_name(a_name)
    b_hit = find_pokemon_by_name(b_name)

    if not a_hit or not b_hit:
        msg = f"포켓몬을 못 찾았어. A='{a_name}', B='{b_name}' 입력을 조금 더 정확히 해봐."
        st.session_state.chat.append({"role": "assistant", "content": msg})
    else:
        _, a_doc = a_hit
        _, b_doc = b_hit

        # (선택) LangChain으로 “해설” 생성
        llm = get_llm()

        context = build_rag_context(a_doc, b_doc)
        mult = type_calc.calc_type_multiplier(move_type, b_doc["meta"].get("types", []), chart)
        label = type_calc.label_by_multiplier(mult)

        prompt_for_llm = f"""
        너는 포켓몬 배틀 해설자야. 아래 [근거]에 있는 정보만 근거로 삼아 설명해.
        모르면 추측하지 말고 "데이터에 없음"이라고 말해.

        [유저 질문]
        {prompt}

        [계산 결과(룰베이스)]
        - 공격 타입: {move_type}
        - 타입 배율: {mult}x
        - 상성 레이블: {label}

        [근거]
        {context}

        요청: 한국어로 5~8줄 정도로, 왜 그런지 근거(타입/스탯)를 섞어서 설명해줘.
        """.strip()

        explain = llm.invoke([HumanMessage(content=prompt_for_llm)]).content


        st.session_state.chat.append({
            "role": "assistant",
            "content": explain
        })

        st.divider()
        render_result(a_doc, b_doc, move_type, mult, label)

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
