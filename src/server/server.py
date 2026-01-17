# server.py
import asyncio
import json
import time
import re
from itertools import combinations
from pathlib import Path
from functools import lru_cache
from typing import Dict, Optional, List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

import utils.type_calc as type_calc

load_dotenv()

# =========================
# RAG/LLM 설정(서버 전용)
# =========================
STORE_DIR = Path("src/rag_store")
INDEX_PATH = STORE_DIR / "pokemon.index"
DOCS_PATH = STORE_DIR / "documents.jsonl"
TYPE_CHART_PATH = Path("src/data/type_chart.json")
MAX_POKEMON = 4

KO_STOPWORDS = {
    "상대로", "에게", "한테", "사용", "기술", "공격", "쓰면", "써", "어때", "어떻게", "좀", "해줘",
    "분석", "결과", "추천", "좋아", "나빠", "가능", "가능해", "말해", "설명",
    "vs", "대",
}


@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


@lru_cache(maxsize=1)
def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_keybert():
    return KeyBERT(model=get_embedder())


@lru_cache(maxsize=1)
def load_faiss_index():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")
    return faiss.read_index(str(INDEX_PATH))


@lru_cache(maxsize=1)
def load_docs():
    if not DOCS_PATH.exists():
        raise FileNotFoundError(f"documents.jsonl not found: {DOCS_PATH}")
    docs = []
    for line in DOCS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            docs.append(json.loads(line))
    return docs


@lru_cache(maxsize=1)
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

    ko_list = sorted(ko_names, key=len, reverse=True)
    en_list = sorted(en_names, key=len, reverse=True)
    return ko_list, en_list


@lru_cache(maxsize=1)
def load_chart():
    if not TYPE_CHART_PATH.exists():
        raise FileNotFoundError(f"type_chart.json not found: {TYPE_CHART_PATH}")
    return type_calc.load_type_chart(TYPE_CHART_PATH)


def detect_lang(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return "ko"
    if re.search(r"[가-힣]", s):
        return "ko"
    return "en"


def normalize_name(s: str) -> str:
    return (s or "").strip()


def extract_pokemon_names_from_text(text: str, max_n: int = MAX_POKEMON) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        return []

    ko_list, en_list = build_name_index()
    found = []
    used = set()

    for name in ko_list:
        if name in raw and name not in used:
            found.append(name)
            used.add(name)
            if len(found) >= max_n:
                return found

    low = raw.lower()
    for en in en_list:
        if len(en) < 4:
            continue
        if re.search(rf"\b{re.escape(en)}\b", low) and en not in used:
            found.append(en)
            used.add(en)
            if len(found) >= max_n:
                return found

    return found


def clean_name_token(name: str, valid_types: set[str]) -> str:
    s = (name or "").strip()

    for ko in type_calc.TYPE_ALIAS_KO.keys():
        s = s.replace(ko, " ")

    for ty in valid_types:
        s = re.sub(rf"\b{re.escape(ty)}\b", " ", s, flags=re.IGNORECASE)

    s = re.sub(r"(상대로|에게|한테|사용|기술|공격|쓰면|써|로)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def refine_input_with_keybert(text: str, valid_types: set[str], top_n: int = 6) -> str:
    raw = (text or "").strip()
    if not raw:
        return raw

    preserved_names = extract_pokemon_names_from_text(raw, max_n=MAX_POKEMON)

    kb = get_keybert()
    keywords = kb.extract_keywords(
        raw,
        keyphrase_ngram_range=(1, 2),
        top_n=top_n,
        use_mmr=True,
        diversity=0.5,
    )

    phrases = [p for (p, s) in keywords if p and s is not None]

    cleaned_tokens = []
    for ph in phrases:
        ph = clean_name_token(ph, valid_types)
        if not ph:
            continue
        for tok in ph.split():
            t = tok.strip()
            if not t:
                continue
            if len(t) <= 1:
                continue
            if t.lower() in KO_STOPWORDS:
                continue
            cleaned_tokens.append(t)

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

    refined = " ".join(final).strip()
    return refined if len(refined) >= 2 else raw


def search_docs(query: str, k: int = 8) -> List[Tuple[float, dict]]:
    embedder = get_embedder()
    index = load_faiss_index()
    docs = load_docs()

    q = embedder.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(q, k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(docs):
            continue
        results.append((float(score), docs[idx]))
    return results


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
        return {"names": names, "move_type": move_type}

    parts = raw.split()
    cleaned = [clean_name_token(p, valid_types) for p in parts]
    cleaned = [c for c in cleaned if c]

    return {"names": cleaned[:MAX_POKEMON], "move_type": move_type}


def build_rag_context_multi(hits: List[Tuple[float, dict]], max_each_text: int = 400, lang: str = "ko") -> str:
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


def ask_llm_only(user_prompt: str, lang: str, memory_text: str = "") -> str:
    llm = get_llm()
    prompt_for_llm = build_llm_prompt(user_prompt, memory_text, lang)
    return llm.invoke([HumanMessage(content=prompt_for_llm)]).content


def build_room_memory(room_chat: List[dict], max_turns: int = 8) -> str:
    chat = (room_chat or [])[-max_turns:]
    lines = []
    for m in chat:
        sender = (m.get("sender") or "").strip() or "unknown"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"user({sender}): {content}" if m.get("role") != "assistant" else f"assistant: {content}")
    return "\n".join(lines)


def bot_generate_reply(user_text: str, room_chat: List[dict]) -> str:
    lang = detect_lang(user_text)

    chart = load_chart()
    valid_types = set(chart.keys())

    memory_text = build_room_memory(room_chat, max_turns=8)

    raw_type = type_calc.guess_move_type_from_text(user_text, valid_types)
    refined_prompt = refine_input_with_keybert(user_text, valid_types)
    parsed = parse_battle_text(refined_prompt, valid_types)
    if not parsed.get("move_type") and raw_type:
        parsed["move_type"] = raw_type

    names = parsed["names"]
    hits = []
    for name in names:
        hit = find_pokemon_by_name(name)
        if hit:
            hits.append(hit)

    if len(hits) == 1:
        context_one = build_rag_context_multi(hits, max_each_text=500, lang=lang)
        llm = get_llm()
        prompt_for_llm = build_llm_prompt(
            user_prompt=user_text,
            memory_text=memory_text,
            lang=lang,
            context=context_one,
        )
        return llm.invoke([HumanMessage(content=prompt_for_llm)]).content

    if len(hits) == 0:
        return ask_llm_only(user_text, lang, memory_text=memory_text)

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
    context_multi = build_rag_context_multi(hits, max_each_text=350, lang=lang)
    prompt_for_llm = build_llm_prompt(
        user_prompt=refined_prompt,
        memory_text=memory_text,
        lang=lang,
        context=context_multi,
        pairs_summary=pairs_summary,
    )
    return llm.invoke([HumanMessage(content=prompt_for_llm)]).content


def answer_with_rag(user_text: str, room_chat: List[dict]) -> str:
    return bot_generate_reply(user_text, room_chat)


# =========================
# 룸/WS 서버
# =========================
class Player:
    def __init__(self, player_id: str, nickname: str):
        self.player_id = player_id
        self.nickname = nickname
        self.ws: Optional[WebSocket] = None


class RoomState:
    def __init__(self, room_id: int):
        self.room_id = room_id
        self.players: List[Player] = []
        self.turn_index: int = 0
        self.chat: List[dict] = []
        self.lock = asyncio.Lock()

    def is_full(self) -> bool:
        return len(self.players) >= 2

    def current_player_id(self) -> Optional[str]:
        if len(self.players) < 2:
            return None  # 2명 미만이면 턴 없음
        return self.players[self.turn_index % len(self.players)].player_id

    def find_player(self, player_id: str) -> Optional[Player]:
        for p in self.players:
            if p.player_id == player_id:
                return p
        return None


rooms: Dict[int, RoomState] = {i: RoomState(i) for i in (1, 2, 3)}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def snapshot_room(room: RoomState) -> dict:
    return {
        "room_id": room.room_id,
        "players": [{"player_id": p.player_id, "nickname": p.nickname} for p in room.players],
        "turn_player_id": room.current_player_id(),
        "chat": room.chat[-200:],
    }


async def broadcast_room(room: RoomState, payload: dict):
    msg = json.dumps(payload, ensure_ascii=False)
    for p in list(room.players):
        if p.ws is None:
            continue
        try:
            await p.ws.send_text(msg)
        except Exception:
            pass


async def _remove_player_from_room(room: RoomState, player_id: str, reason: str) -> bool:
    p = room.find_player(player_id)
    if not p:
        return False

    nickname = p.nickname

    idx_removed = None
    for i, x in enumerate(room.players):
        if x.player_id == player_id:
            idx_removed = i
            break

    cur_idx = room.turn_index % len(room.players) if room.players else 0

    if p.ws:
        try:
            await p.ws.close()
        except Exception:
            pass
        p.ws = None

    room.players = [x for x in room.players if x.player_id != player_id]

    room.chat.append({
        "role": "system",
        "sender": "system",
        "content": f"{nickname} {reason}",
        "ts": time.time(),
    })

    if len(room.players) < 2:
        room.turn_index = 0
    else:
        if idx_removed is not None and idx_removed < cur_idx:
            cur_idx -= 1
        room.turn_index = cur_idx % len(room.players)

    return True


@app.get("/rooms")
async def get_rooms():
    data = []
    for r in rooms.values():
        data.append({
            "room_id": r.room_id,
            "count": len(r.players),
            "is_full": r.is_full(),
            "turn_player_id": r.current_player_id(),
        })
    return {"rooms": data}


@app.get("/rooms/{room_id}")
async def get_room(room_id: int):
    room = rooms.get(room_id)
    if not room:
        raise HTTPException(404, "room not found")
    return {"room": snapshot_room(room)}


@app.post("/rooms/{room_id}/join")
async def join_room(room_id: int, player_id: str, nickname: str):
    room = rooms.get(room_id)
    if not room:
        raise HTTPException(404, "room not found")

    async with room.lock:
        if room.find_player(player_id):
            return {"room": snapshot_room(room)}
        if room.is_full():
            raise HTTPException(409, "room is full")

        room.players.append(Player(player_id, nickname))
        room.chat.append({"role": "system", "sender": "system", "content": f"{nickname} 입장", "ts": time.time()})

        if len(room.players) == 2:
            room.turn_index = 0
            room.chat.append({
                "role": "system",
                "sender": "system",
                "content": f"게임 시작! 선턴: {room.players[0].nickname}",
                "ts": time.time(),
            })

        payload = {"type": "room_update", "room": snapshot_room(room)}

    await broadcast_room(room, payload)
    return {"room": payload["room"]}


@app.post("/rooms/{room_id}/leave")
async def leave_room(room_id: int, player_id: str):
    room = rooms.get(room_id)
    if not room:
        raise HTTPException(404, "room not found")

    async with room.lock:
        removed = await _remove_player_from_room(room, player_id, reason="퇴장")
        payload = {"type": "room_update", "room": snapshot_room(room)}

    if removed:
        await broadcast_room(room, payload)
    return {"ok": True}


@app.websocket("/ws/rooms/{room_id}/{player_id}")
async def ws_room(websocket: WebSocket, room_id: int, player_id: str):
    room = rooms.get(room_id)
    if not room:
        await websocket.close(code=1008)
        return

    await websocket.accept()

    async with room.lock:
        p = room.find_player(player_id)
        if not p:
            await websocket.send_text(json.dumps({"type": "notice", "code": "NEED_JOIN", "message": "join 먼저 해야됨"}, ensure_ascii=False))
            await websocket.close(code=1008)
            return
        p.ws = websocket
        snap = snapshot_room(room)

    await websocket.send_text(json.dumps({"type": "room_snapshot", "room": snap}, ensure_ascii=False))
    await broadcast_room(room, {"type": "room_update", "room": snap})

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except Exception:
                continue

            if data.get("type") != "chat":
                continue

            user_text = (data.get("content") or "").strip()
            if not user_text:
                continue

            async with room.lock:
                if len(room.players) < 2:
                    await websocket.send_text(json.dumps({"type": "notice", "code": "WAITING", "message": "상대방 입장 대기 중"}, ensure_ascii=False))
                    continue

                if room.current_player_id() != player_id:
                    await websocket.send_text(json.dumps({"type": "notice", "code": "NOT_YOUR_TURN", "message": "상대의 턴입니다."}, ensure_ascii=False))
                    continue

                player = room.find_player(player_id)
                if not player:
                    await websocket.send_text(json.dumps({"type": "notice", "code": "NOT_IN_ROOM", "message": "방에 없음"}, ensure_ascii=False))
                    continue

                room.chat.append({"role": "user", "sender": player.nickname, "content": user_text, "ts": time.time()})
                chat_copy = list(room.chat)

            try:
                assistant_text = await asyncio.to_thread(answer_with_rag, user_text, chat_copy)
            except Exception as e:
                assistant_text = f"(assistant) 처리 중 에러: {e}"

            async with room.lock:
                room.chat.append({"role": "assistant", "sender": "assistant", "content": assistant_text, "ts": time.time()})
                if len(room.players) >= 2:
                    room.turn_index = (room.turn_index + 1) % len(room.players)
                payload = {"type": "room_update", "room": snapshot_room(room)}

            await broadcast_room(room, payload)

    except WebSocketDisconnect:
        async with room.lock:
            await _remove_player_from_room(room, player_id, reason="연결 끊김")
            payload = {"type": "room_update", "room": snapshot_room(room)}
        await broadcast_room(room, payload)
