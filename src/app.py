import json
import re
from pathlib import Path

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer


STORE_DIR = Path("rag_store")
INDEX_PATH = STORE_DIR / "pokemon.index"
DOCS_PATH = STORE_DIR / "documents.jsonl"
TYPE_CHART_PATH = Path("type_chart_full.json")


def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def load_type_chart(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calc_type_multiplier(atk_type: str, def_types: list[str], chart: dict) -> float:
    atk = (atk_type or "").strip().lower()
    mult = 1.0
    for dt in def_types:
        if not dt:
            continue
        d = str(dt).strip().lower()
        mult *= float(chart.get(atk, {}).get(d, 1.0))
    return mult


def calc_stats_advantage(a_stats: dict, b_stats: dict) -> dict:
    speed_diff = (a_stats.get("spe", 0) - b_stats.get("spe", 0))
    phys_edge = (a_stats.get("atk", 0) - b_stats.get("def", 0))
    spec_edge = (a_stats.get("spa", 0) - b_stats.get("spd", 0))

    speed_score = clamp(speed_diff / 30.0)
    phys_score = clamp(phys_edge / 50.0)
    spec_score = clamp(spec_edge / 50.0)
    offense_score = max(phys_score, spec_score)

    return {
        "speed_diff": int(speed_diff),
        "phys_edge": int(phys_edge),
        "spec_edge": int(spec_edge),
        "speed_score": float(speed_score),
        "phys_score": float(phys_score),
        "spec_score": float(spec_score),
        "offense_score": float(offense_score),
        "favored_offense": "special" if spec_score >= phys_score else "physical",
    }


def pick_adv_label(total_score: float) -> str:
    if total_score >= 1.2:
        return "매우 유리"
    if total_score >= 0.4:
        return "유리"
    if total_score > -0.4:
        return "비슷"
    if total_score > -1.2:
        return "불리"
    return "매우 불리"


@st.cache_resource
def load_model():
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
def load_chart():
    if not TYPE_CHART_PATH.exists():
        raise FileNotFoundError(f"type_chart_full.json not found: {TYPE_CHART_PATH}")
    return load_type_chart(TYPE_CHART_PATH)


def search_docs(query: str, k: int = 8):
    model = load_model()
    index = load_faiss_index()
    docs = load_docs()

    q = model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(q, k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(docs):
            continue
        results.append((float(score), docs[idx]))
    return results


def safe_types(meta: dict) -> list[str]:
    if isinstance(meta.get("types"), list) and meta["types"]:
        return [t for t in meta["types"] if t]
    return []


def normalize_name(s: str) -> str:
    return (s or "").strip()


def find_pokemon_by_name(name: str):
    """
    1) documents.jsonl에서 korean_name/english_name 정확매칭 우선
    2) 없으면 faiss로 유사검색해서 top1
    """
    name = normalize_name(name)
    if not name:
        return None

    docs = load_docs()
    low = name.lower()

    # exact match (korean / english)
    for d in docs:
        m = d.get("meta", {})
        if (m.get("korean_name") or "").strip() == name:
            return 1.0, d
        if (m.get("english_name") or "").strip().lower() == low:
            return 1.0, d

    # fallback: faiss
    hits = search_docs(name, k=5)
    return hits[0] if hits else None


TYPE_ALIAS_KO = {
    "노말": "normal",
    "불": "fire",
    "불꽃": "fire",
    "물": "water",
    "전기": "electric",
    "풀": "grass",
    "얼음": "ice",
    "격투": "fighting",
    "독": "poison",
    "땅": "ground",
    "비행": "flying",
    "에스퍼": "psychic",
    "벌레": "bug",
    "바위": "rock",
    "고스트": "ghost",
    "드래곤": "dragon",
    "악": "dark",
    "강철": "steel",
    "페어리": "fairy",
}

def guess_move_type_from_text(text: str, valid_types: set[str]) -> str | None:
    t = text.strip().lower()
    # 한글 타입 별칭 우선
    for k, v in TYPE_ALIAS_KO.items():
        if k in text and v in valid_types:
            return v
    # 영문 타입 직접 언급
    for ty in valid_types:
        if re.search(rf"\b{re.escape(ty)}\b", t):
            return ty
    return None


def parse_battle_text(text: str, valid_types: set[str]) -> dict:
    """
    LLM 없이도 일단 돌아가게 만드는 정규식 파서.
    사용자가:
      - "리자몽이 이상해꽃 상대로 불꽃 기술 쓰면?"
      - "피카츄 vs 꼬부기 전기"
    같은 형태로 넣는 걸 가정.
    """
    raw = text.strip()
    if not raw:
        return {"a": "", "b": "", "move_type": None}

    # A vs B
    m = re.search(r"(.+?)\s*(?:vs|VS|대)\s*(.+)", raw)
    if m:
        a = m.group(1).strip()
        b = m.group(2).strip()
        move_type = guess_move_type_from_text(raw, valid_types)
        return {"a": a, "b": b, "move_type": move_type}

    # "A가 B" or "A이/가 B를"
    m = re.search(r"(.+?)(?:이|가)\s*(.+?)(?:를|을|상대로|에게|한테)", raw)
    if m:
        a = m.group(1).strip()
        b = m.group(2).strip()
        move_type = guess_move_type_from_text(raw, valid_types)
        return {"a": a, "b": b, "move_type": move_type}

    # 그냥 공백 2개 토큰이면 A B로
    parts = raw.split()
    a = parts[0] if len(parts) >= 1 else ""
    b = parts[1] if len(parts) >= 2 else ""
    move_type = guess_move_type_from_text(raw, valid_types)
    return {"a": a, "b": b, "move_type": move_type}


def render_result(a_doc, b_doc, atk_type: str, chart: dict):
    a_meta = a_doc.get("meta", {})
    b_meta = b_doc.get("meta", {})

    a_types = safe_types(a_meta)
    b_types = safe_types(b_meta)

    a_stats = a_meta.get("stats", {}) if isinstance(a_meta.get("stats", {}), dict) else {}
    b_stats = b_meta.get("stats", {}) if isinstance(b_meta.get("stats", {}), dict) else {}

    type_mult = calc_type_multiplier(atk_type, b_types, chart)
    stats_adv = calc_stats_advantage(a_stats, b_stats)

    if type_mult == 0:
        type_score = -999.0
    else:
        type_score = float(np.log2(type_mult))

    total_score = 0.65 * type_score + 0.20 * stats_adv["speed_score"] + 0.15 * stats_adv["offense_score"]
    label = "무효(0배)라 거의 불리" if type_mult == 0 else pick_adv_label(total_score)

    st.subheader("결과 요약")
    st.markdown(
        f"""
- **A:** {a_meta.get("korean_name") or a_meta.get("english_name") or "?"} ({a_meta.get("english_name","?")})
- **B:** {b_meta.get("korean_name") or b_meta.get("english_name") or "?"} ({b_meta.get("english_name","?")})
- **공격 타입:** `{atk_type}`
- **타입 배율(룰):** **{type_mult}x**
- **스탯 관점:** 스피드 차이 **{stats_adv["speed_diff"]:+}**, 유리한 공격 성향 **{stats_adv["favored_offense"]}**
- **종합 판단:** **{label}**
"""
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### A 근거(RAG)")
        st.code(a_doc.get("text", "")[:1500])
        st.json({"meta": a_meta})

    with c2:
        st.markdown("### B 근거(RAG)")
        st.code(b_doc.get("text", "")[:1500])
        st.json({"meta": b_meta})


# ---------------- UI ----------------
st.set_page_config(page_title="Pokemon Battle Chat (RAG+Rules)", layout="wide")
st.title("포켓몬 배틀 시뮬레이션 챗봇")
st.caption("입력 텍스트에서 포켓몬을 추출하고(파싱), 계산은 룰베이스로만 처리함. RAG는 설명/근거만.")

chart = load_chart()
valid_types = set(chart.keys())
all_types = sorted(valid_types)

if "chat" not in st.session_state:
    st.session_state.chat = []

prompt = st.chat_input("예) 리자몽이 이상해꽃 상대로 불꽃 기술 쓰면 어때? / 피카츄 vs 꼬부기 전기")

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})

    parsed = parse_battle_text(prompt, valid_types)
    a_name = parsed["a"]
    b_name = parsed["b"]
    move_type = parsed["move_type"] or "normal"  # 타입 못 찾으면 일단 노말

    a_hit = find_pokemon_by_name(a_name)
    b_hit = find_pokemon_by_name(b_name)

    if not a_hit or not b_hit:
        msg = f"포켓몬을 못 찾았어. A='{a_name}', B='{b_name}' 입력을 조금 더 정확히 해봐."
        st.session_state.chat.append({"role": "assistant", "content": msg})
    else:
        _, a_doc = a_hit
        _, b_doc = b_hit

        # 결과는 UI에서 바로 출력(assistant 메시지는 요약만)
        st.session_state.chat.append({
            "role": "assistant",
            "content": f"A={a_name}, B={b_name}, 타입={move_type} 로 계산했어. 아래 결과 확인해봐."
        })

        st.divider()
        render_result(a_doc, b_doc, move_type, chart)

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
