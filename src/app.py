# app.py
from itertools import combinations
import json
import re
from pathlib import Path

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
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
MAX_POKEMON = 4
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
def get_keybert():
    return KeyBERT(model=get_embedder())

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

KO_STOPWORDS = {
    "상대로","에게","한테","사용","기술","공격","쓰면","써","어때","어떻게","좀","해줘",
    "분석","결과","추천","좋아","나빠","가능","가능해","말해","설명",
    "vs","대",
}

def refine_input_with_keybert(text: str, valid_types: set[str], top_n: int = 6) -> str:
    raw = (text or "").strip()
    if not raw:
        return raw

    # 1) 포켓몬 이름은 기존 로직으로 먼저 확보(보존용)
    preserved_names = extract_pokemon_names_from_text(raw, max_n=MAX_POKEMON)

    # 2) KeyBERT로 키워드 추출
    kb = get_keybert()
    keywords = kb.extract_keywords(
        raw,
        keyphrase_ngram_range=(1, 2),
        top_n=top_n,
        use_mmr=True,        # 다양성 조금 확보
        diversity=0.5,
    )

    # keywords: [(phrase, score), ...]
    phrases = [p for (p, s) in keywords if p and s is not None]

    # 3) 정제: 타입 단어/조사/잡단어 제거 + 너무 짧은 토큰 제거
    cleaned_tokens = []
    for ph in phrases:
        ph = clean_name_token(ph, valid_types)  # 너가 이미 만들어둔 “타입단어/조사 제거기”
        if not ph:
            continue

        # 공백 split 후 추가 필터링
        for tok in ph.split():
            t = tok.strip()
            if not t:
                continue
            if len(t) <= 1:
                continue
            if t.lower() in KO_STOPWORDS:
                continue
            cleaned_tokens.append(t)

    # 4) 포켓몬 이름/타입 키워드는 우선순위로 앞에 넣기
    # (타입은 guess_move_type이 원문/정제문 둘 다에서 잘 잡게끔)
    final = []
    used = set()

    for n in preserved_names:
        if n not in used:
            final.append(n)
            used.add(n)

    for t in cleaned_tokens:
        if t not in used:
            final.append(t)
            used.add(t)

    # 너무 짧아지면 원문 유지(정제 실패 방어)
    refined = " ".join(final).strip()
    return refined if len(refined) >= 2 else raw

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

def extract_pokemon_names_from_text(text: str, max_n: int = MAX_POKEMON) -> list[str]:
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
        return {"names": [], "move_type": None}

    move_type = type_calc.guess_move_type_from_text(raw, valid_types)

    names = extract_pokemon_names_from_text(raw, max_n=MAX_POKEMON)
    if names:
        return {
            "names": names,
            "move_type": move_type
        }

    # fallback
    parts = raw.split()
    cleaned = [clean_name_token(p, valid_types) for p in parts]
    cleaned = [c for c in cleaned if c]

    return {
        "names": cleaned[:MAX_POKEMON],
        "move_type": move_type
    }


def render_results(results: list[dict]):
    st.divider()
    st.subheader("포켓몬 상성 분석(자속 기준, 양방향)")

    for r in results:
        atk_name = r["attacker_doc"]["meta"].get("korean_name") or r["attacker_doc"]["meta"].get("english_name") or ""
        def_name = r["defender_doc"]["meta"].get("korean_name") or r["defender_doc"]["meta"].get("english_name") or ""

        st.markdown(f"### {atk_name} ➜ {def_name}")
        st.write(f"자속 타입 선택: {r['move_type']}")
        st.write(f"타입 배율: {r['type_mult']:.2f}x")
        st.write(f"자속(STAB): {r['stab']:.1f}x")
        st.write(f"최종 배율: {r['mult']:.2f}x")
        st.write(f"레이블(타입상성): {r['label']}")
        st.write("---")


def ask_llm_only(user_prompt: str, lang: str) -> str:
    llm = get_llm()
    memory_text = build_chat_memory(max_turns=6, lang=lang)
    prompt_for_llm = build_llm_prompt(user_prompt, memory_text, lang)
    return llm.invoke([HumanMessage(content=prompt_for_llm)]).content

def build_rag_context_multi(hits: list[tuple[float, dict]], max_each_text: int = 400, lang: str = "ko") -> str:
    lines = []
    for score, doc in hits:
        meta = doc.get("meta", {})
        name = meta.get("korean_name") or meta.get("english_name") or ""
        types = meta.get("types", [])
        stats = meta.get("stats", {})
        text = (doc.get("text", "") or "")[:max_each_text]

        if lang == "en":
            lines.append(f"""
[EVIDENCE: Pokemon]
Name: {name}
Types: {types}
Stats: {stats}
Description: {text}
""".strip())
        else:
            lines.append(f"""
[포켓몬 근거]
이름: {name}
타입: {types}
스탯: {stats}
설명: {text}
""".strip())
    return "\n\n".join(lines).strip()


def build_chat_memory(max_turns: int = 6, lang: str = "ko") -> str:
    chat = st.session_state.get("chat", [])
    if not chat:
        return ""

    recent = chat[-max_turns:]
    lines = []
    for m in recent:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue

        if lang == "en":
            prefix = "User" if role == "user" else "Assistant"
        else:
            prefix = "유저" if role == "user" else "assistant"

        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def detect_lang(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return "ko"
    # 한글 포함이면 한국어로
    if re.search(r"[가-힣]", s):
        return "ko"
    return "en"

def build_llm_prompt(user_prompt: str, memory_text: str, lang: str, context: str = "", pairs_summary: str = "") -> str:
    if lang == "en":
        base = """
You are a Pokémon battle commentator.
Use ONLY the provided [EVIDENCE]. If you don't know, say "Not in the data" and do not guess.
""".strip()
        mem_label = "[CHAT MEMORY (recent)]"
        user_label = "[USER QUESTION]"
        calc_label = "[CALC RESULTS (pairs)]"
        evidence_label = "[EVIDENCE]"
        request = "Request: Answer in English in about 5-8 lines."
    else:
        base = """
너는 포켓몬 배틀 해설자야.
아래 [근거]만 근거로 설명해.
모르면 추측하지 말고 "데이터에 없음"이라고 말해.
""".strip()
        mem_label = "[대화 메모리(최근)]"
        user_label = "[유저 질문]"
        calc_label = "[계산 결과(여러 쌍)]"
        evidence_label = "[근거]"
        request = "요청: 한국어로 5~8줄 정도로 답해줘."

    parts = [base, "", mem_label, memory_text or "", "", user_label, user_prompt or ""]

    if pairs_summary:
        parts += ["", calc_label, pairs_summary]

    if context:
        parts += ["", evidence_label, context]

    parts += ["", request]
    return "\n".join(parts).strip()


# ---------------- UI ----------------
st.set_page_config(page_title="Pokemon Battle Chat (RAG+Rules)", layout="wide")
st.title("포켓몬 배틀 시뮬레이션 챗봇")

chart = load_chart()
valid_types = set(chart.keys())

if "chat" not in st.session_state:
    st.session_state.chat = []

if "memo_cache" not in st.session_state:
    st.session_state.memo_cache = {}

prompt = st.chat_input("예) 리자몽이 이상해꽃 상대로 불꽃 기술 쓰면 어때? / 피카츄 vs 꼬부기 전기")

if prompt:
    lang = detect_lang(prompt)
    st.session_state.chat.append({"role": "user", "content": prompt})
    cache_key = re.sub(r"\s+", " ", prompt.strip())

    if cache_key in st.session_state.memo_cache:
        cached = st.session_state.memo_cache[cache_key]
        refined_prompt = cached["refined_prompt"]
        parsed = cached["parsed"]
        hits = cached["hits"]
        if not parsed.get("move_type"):
            raw_type = type_calc.guess_move_type_from_text(prompt, valid_types)
            if raw_type:
                parsed["move_type"] = raw_type
                st.session_state.memo_cache[cache_key]["parsed"] = parsed
    else:

        raw_type  = type_calc.guess_move_type_from_text(prompt, valid_types)

        refined_prompt = refine_input_with_keybert(prompt, valid_types)
        parsed = parse_battle_text(refined_prompt, valid_types)
        if not parsed.get("move_type") and raw_type :
            parsed["move_type"] = raw_type 
        names = parsed["names"]
        hits = []
        for name in names:
            hit = find_pokemon_by_name(name)
            if hit:
                hits.append(hit)

        # ✅ 캐시 저장
        MAX_CACHE = 50
        if len(st.session_state.memo_cache) >= MAX_CACHE:
            oldest_key = next(iter(st.session_state.memo_cache))
            del st.session_state.memo_cache[oldest_key]

        st.session_state.memo_cache[cache_key] = {
            "refined_prompt": refined_prompt,
            "parsed": parsed,
            "hits": hits,
        }
    
    if len(hits) == 1:
        context_one = build_rag_context_multi(hits, max_each_text=500, lang=lang)
        llm = get_llm()
        memory_text = build_chat_memory(max_turns=6, lang=lang)

        prompt_for_llm = build_llm_prompt(
            user_prompt=prompt,
            memory_text=memory_text,
            lang=lang,
            context=context_one
        )
        explain = llm.invoke([HumanMessage(content=prompt_for_llm)]).content
        st.session_state.chat.append({"role": "assistant", "content": explain})
    elif len(hits) == 0:
        explain = ask_llm_only(prompt, lang)
        st.session_state.chat.append({"role": "assistant", "content": explain})
    else:
        results = []
        user_move_type = (parsed.get("move_type") or "").strip().lower() or None
        for (a_score, a_doc), (b_score, b_doc) in combinations(hits, 2):
            if user_move_type:
                ab = type_calc.calc_given_type_attack(a_doc, b_doc, user_move_type, chart)
                ba = type_calc.calc_given_type_attack(b_doc, a_doc, user_move_type, chart)
            else:
                ab = type_calc.calc_best_stab_attack(a_doc, b_doc, chart)
                ba = type_calc.calc_best_stab_attack(b_doc, a_doc, chart)

            results.append({"attacker_doc": a_doc, "defender_doc": b_doc, **ab})
            results.append({"attacker_doc": b_doc, "defender_doc": a_doc, **ba})


        render_results(results)

        summary_lines = []
        for r in results:
            atk_name = r["attacker_doc"]["meta"].get("korean_name") or r["attacker_doc"]["meta"].get("english_name") or ""
            def_name = r["defender_doc"]["meta"].get("korean_name") or r["defender_doc"]["meta"].get("english_name") or ""
            stab_txt = "자속" if r["stab"] > 1.0 else "비자속"
            summary_lines.append(
                f"- {atk_name}→{def_name}: {r['move_type']} ({stab_txt}), {r['type_mult']:.2f}x×{r['stab']:.1f}={r['mult']:.2f}x ({r['label']})"
            )
        pairs_summary = "\n".join(summary_lines)


        llm = get_llm()
        memory_text = build_chat_memory(max_turns=6, lang=lang)
        context_multi = build_rag_context_multi(hits, max_each_text=350, lang=lang)
        prompt_for_llm = build_llm_prompt(
            user_prompt=refined_prompt,
            memory_text=memory_text,
            lang=lang,
            context=context_multi,
            pairs_summary=pairs_summary
        )
        explain = llm.invoke([HumanMessage(content=prompt_for_llm)]).content
        st.session_state.chat.append({"role": "assistant", "content": explain})

        st.divider()

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
