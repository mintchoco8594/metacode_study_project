# server.py
import asyncio
import json
import time
import re
import random
from pathlib import Path
from functools import lru_cache
from typing import Dict, Optional, List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from src.utils import type_calc

load_dotenv()

# =========================
# RAG/LLM 설정(서버 전용)
# =========================
STORE_DIR = Path("src/rag_store")
INDEX_PATH = STORE_DIR / "pokemon.index"
DOCS_PATH = STORE_DIR / "documents.jsonl"
TYPE_CHART_PATH = Path("src/data/type_chart.json")
MAX_POKEMON = 1



@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


@lru_cache(maxsize=1)
def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


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


def build_llm_prompt(
    user_prompt: str,
    memory_text: str,
    lang: str,
    context: str = "",
    pairs_summary: str = "",
) -> str:
    if lang == "en":
        base = """
You are a Pokémon battle commentator.

Rules (MANDATORY):
- Use ONLY information explicitly written in [CALC RESULTS] and [EVIDENCE].
- Do NOT invent moves, turns, status effects, abilities, items, accuracy, critical hits, or battle flow.
- Do NOT describe actions like "used X move" or "attacked first".
- If something is not written in the evidence, say exactly: "Not in the data".
- The winner is already determined in the evidence. Do NOT change it.
""".strip()

        request = """
Output format (STRICT):
Winner or Draw (1 line)
Comment: Comment each pokemon's type and use type multiplier and score from [CALC RESULTS] (4~6 lines)
Total length: 5~8 lines.
""".strip()

        mem_label = "[CHAT MEMORY]"
        user_label = "[USER REQUEST]"
        calc_label = "[CALC RESULTS]"
        evidence_label = "[EVIDENCE]"

    else:
        base = """
너는 포켓몬 배틀 해설자야.

규칙(반드시 지켜):
- 오직 [계산 결과]와 [근거]에 **명시된 정보만** 사용해.
- 기술 사용, 턴 순서, 상태이상, 특성, 아이템, 급소, 명중/회피 같은
  배틀 연출을 절대 만들어내지 마.
- 승자, 타입 배율, score는 반드시 [계산 결과]에 있는 문장만 인용해.
- 근거에 없는 내용은 반드시 정확히 "데이터에 없음"이라고 말해.
- 승자는 이미 확정되어 있으니 바꾸거나 추측하지 마.
""".strip()

        request = """
출력 형식(엄격):
승자 또는 무승부 (1줄)
해설: 각 포켓몬의 타입, 타입 배율 + score 근거 설명 (4~6줄)
총 5~8줄.
""".strip()

    mem_label = "[대화 메모리]"
    user_label = "[유저 요청]"
    calc_label = "[계산 결과]"
    evidence_label = "[근거]"

    parts = [
        base,
        "",
        mem_label,
        memory_text or "",
        "",
        user_label,
        user_prompt or "",
    ]

    if pairs_summary:
        parts += ["", calc_label, pairs_summary]

    if context:
        parts += ["", evidence_label, context]

    parts += ["", request]

    return "\n".join(parts).strip()



# =========================
# 룸/WS 서버
# =========================
class Player:
    def __init__(self, player_id: str, nickname: str):
        self.player_id = player_id
        self.nickname = nickname
        self.ws: Optional[WebSocket] = None
        self.picked: bool = False
        self.pokemon_name: Optional[str] = None
        self.pokemon_doc: Optional[dict] = None
        self.lives: int = 3
        self.auto_play: bool = False

class RoomState:
    def __init__(self, room_id: int):
        self.room_id = room_id
        self.players: List[Player] = []
        self.turn_index: int = 0
        self.chat: List[dict] = []
        self.lock = asyncio.Lock()
        self.phase: str = "pick"          # pick | battle_running | ended
        self.winner_player_id: Optional[str] = None
        self.auto_play: bool = False
        self.used_bot_names: set[str] = set()
        self.battle_task: Optional[asyncio.Task] = None
        self.battle_nonce: int = 0

    def is_full(self) -> bool:
        return len(self.players) >= 2

    def current_player_id(self) -> Optional[str]:
        if len(self.players) < 2:
            return None
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
        "phase": room.phase,
        "winner_player_id": room.winner_player_id,
        "auto_play": room.auto_play,
        "players": [
            {
                "player_id": p.player_id,
                "nickname": p.nickname,
                "picked": p.picked,
                "pokemon_name": (p.pokemon_name if room.phase != "pick" else None),
                "lives": p.lives,
                "auto_play": p.auto_play,
            }
            for p in room.players
        ],
        "turn_player_id": room.current_player_id(),
        "chat": room.chat[-200:],
    }


def get_offense_stat(doc: dict) -> float:
    stats = (doc.get("meta", {}) or {}).get("stats", {}) or {}
    atk = float(stats.get("attack", 0) or 0)
    spa = float(stats.get("sp_attack", 0) or 0)
    return max(atk, spa, 1.0)


def simulate_battle(p1_doc: dict, p2_doc: dict, chart: dict) -> dict:
    a_to_b = type_calc.calc_best_stab_attack(p1_doc, p2_doc, chart)
    b_to_a = type_calc.calc_best_stab_attack(p2_doc, p1_doc, chart)

    a_stat = get_offense_stat(p1_doc)
    b_stat = get_offense_stat(p2_doc)

    a_score = float(a_to_b["mult"]) * a_stat
    b_score = float(b_to_a["mult"]) * b_stat

    if abs(a_score - b_score) < 1e-6:
        winner = "draw"
    else:
        winner = "p1" if a_score > b_score else "p2"

    return {
        "a_to_b": a_to_b,
        "b_to_a": b_to_a,
        "a_score": a_score,
        "b_score": b_score,
        "winner": winner,
    }

BOT_PREFIX = "BOT-"

def is_bot(player_id: str) -> bool:
    return (player_id or "").startswith(BOT_PREFIX)

def calc_bst(doc: dict) -> int:
    stats = (doc.get("meta", {}) or {}).get("stats", {}) or {}
    keys = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
    return int(sum(int(stats.get(k, 0) or 0) for k in keys))

@lru_cache(maxsize=1)
def build_stat_pool(stat_limit: int) -> List[dict]:
    docs = load_docs()
    pool = []
    for d in docs:
        if calc_bst(d) >= stat_limit:
            pool.append(d)
    return pool

def room_has_bot(room: RoomState) -> bool:
    return any(is_bot(p.player_id) for p in room.players)

def room_in_progress(room: RoomState) -> bool:
    return room_has_bot(room) or (len(room.players) >= 2)


async def broadcast_room(room: RoomState, payload: dict):
    msg = json.dumps(payload, ensure_ascii=False)
    for p in list(room.players):
        if p.ws is None:
            continue
        try:
            await asyncio.wait_for(p.ws.send_text(msg), timeout=1.0)
        except Exception:
            pass


async def _remove_player_from_room(room: RoomState, player_id: str, reason: str) -> bool:
    p = room.find_player(player_id)
    if not p:
        return False

    nickname = p.nickname

    # ✅ 진행 중 배틀 무효화/취소 (race 방지)
    room.battle_nonce += 1
    t = room.battle_task
    room.battle_task = None
    if t:
        try:
            t.cancel()
        except Exception:
            pass

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
        # ✅ 사람이 1명 이하가 되면 진행중 상태 풀어줌
        if room.phase == "battle_running":
            room.phase = "pick"
            room.winner_player_id = None
    else:
        if idx_removed is not None and idx_removed < cur_idx:
            cur_idx -= 1
        room.turn_index = cur_idx % len(room.players)

    return True


async def bot_pick_random(room: RoomState, bot: Player):
    pool = build_stat_pool(430)
    if not pool:
        pool = load_docs()

    candidates = []
    for doc in pool:
        meta = doc.get("meta", {}) or {}
        name = (meta.get("korean_name") or meta.get("english_name") or "").strip()
        if not name:
            continue
        if name not in room.used_bot_names:
            candidates.append((name, doc))

    if candidates:
        name, doc = random.choice(candidates)
    else:
        doc = random.choice(pool)
        meta = doc.get("meta", {}) or {}
        name = (meta.get("korean_name") or meta.get("english_name") or "UNKNOWN").strip()

    if name and name != "UNKNOWN":
        room.used_bot_names.add(name)

    bot.picked = True
    bot.pokemon_name = name
    bot.pokemon_doc = doc

    room.chat.append({
        "role": "system",
        "sender": "system",
        "content": f"{bot.nickname} 선택 완료!",
        "ts": time.time(),
    })

    room.turn_index = (room.turn_index + 1) % len(room.players)


async def run_bot_until_human_turn(room: RoomState):
    """
    ✅ 싱글플레이일 때:
    - 봇 턴이면 pick 단계에서 자동 선택
    - 인간 턴이 오면 멈춤
    - battle은 기존 ws_room 흐름에서 자동 진행되니까 pick만 처리해도 충분
    """
    for _ in range(5):  # 안전장치(무한루프 방지)
        await asyncio.sleep(0.15)

        async with room.lock:
            if not room.auto_play:
                return
            if len(room.players) < 2:
                return
            if room.phase in ("battle_running", "ended"):
                return

            cur_id = room.current_player_id()
            bot = next((p for p in room.players if p.player_id == cur_id and p.auto_play), None)

            # 봇 턴 아니면 종료
            if bot is None:
                return

            # pick 단계에서만 봇이 행동
            if room.phase == "pick":
                # 이미 골랐으면 넘김(이상상태 방어)
                if bot.picked:
                    room.turn_index = (room.turn_index + 1) % len(room.players)
                else:
                    await bot_pick_random(room, bot)

                battle_job = make_battle_job(room)  # ✅ bot이 마지막 픽이면 여기서 battle_running + battle_job 생성됨
                payload = {"type":"room_update", "room": snapshot_room(room)}
                if battle_job:
                    schedule_battle(room, battle_job)
            else:
                return

        await broadcast_room(room, payload)


def make_battle_job(room: RoomState) -> Optional[dict]:
    if len(room.players) < 2:
        return None
    if not all(p.picked for p in room.players):
        return None

    p1, p2 = room.players[0], room.players[1]
    if not p1.pokemon_doc or not p2.pokemon_doc:
        return None

    chart = load_chart()
    sim = simulate_battle(p1.pokemon_doc, p2.pokemon_doc, chart)

    if sim["winner"] == "p1":
        winner_player_id, loser_player_id = p1.player_id, p2.player_id
        win_name, lose_name = p1.nickname, p2.nickname
    elif sim["winner"] == "p2":
        winner_player_id, loser_player_id = p2.player_id, p1.player_id
        win_name, lose_name = p2.nickname, p1.nickname
    else:
        winner_player_id = None
        loser_player_id = None
        win_name = "무승부"
        lose_name = "무승부"

    hits = [(1.0, p1.pokemon_doc), (1.0, p2.pokemon_doc)]
    context = build_rag_context_multi(hits, max_each_text=350, lang="ko")

    result_evidence = "\n".join([
        "[확정 결과]",
        f"승자: {win_name}",
        f"패자: {lose_name}" if loser_player_id else "패자: 없음(무승부)",
        f"{p1.pokemon_name}→{p2.pokemon_name}: {sim['a_to_b']['move_type']} {sim['a_to_b']['mult']:.2f}x ({sim['a_to_b']['label']}) score={sim['a_score']:.1f}",
        f"{p2.pokemon_name}→{p1.pokemon_name}: {sim['b_to_a']['move_type']} {sim['b_to_a']['mult']:.2f}x ({sim['b_to_a']['label']}) score={sim['b_score']:.1f}",
    ])

    # ✅ 여기서만 battle_running 전환 + 선택완료 로그 1번만 남김
    room.phase = "battle_running"
    room.chat.append({
        "role": "system",
        "sender": "system",
        "content": f"선택 완료! {p1.nickname}: {p1.pokemon_name} / {p2.nickname}: {p2.pokemon_name}",
        "ts": time.time(),
    })

    return {
        "winner_player_id": winner_player_id,
        "loser_player_id": loser_player_id,
        "win_name": win_name,
        "lose_name": lose_name,
        "context": context.strip(),
        "pairs_summary": result_evidence,
        "user_prompt": "위 [계산 결과]와 [근거]만 사용해서 5~8줄로 해설해. 근거에 없는 배틀 전개/기술/턴/상태이상/아이템/특성은 절대 말하지 마.",
    }


def schedule_battle(room: RoomState, battle_job: dict):
    # ✅ room.lock 안에서만 호출한다고 가정
    if room.battle_task is not None:
        return

    room.battle_nonce += 1
    nonce = room.battle_nonce

    task = asyncio.create_task(run_battle_job(room, battle_job, nonce))
    room.battle_task = task

    # ✅ cancel/예외로 run_battle_job이 cleanup 못 하고 죽어도 battle_task 정리되게 안전장치
    def _done_callback(t: asyncio.Task):
        async def _cleanup():
            async with room.lock:
                if room.battle_task is t:
                    room.battle_task = None
        asyncio.create_task(_cleanup())

    task.add_done_callback(_done_callback)


async def run_battle_job(room: RoomState, battle_job: dict, nonce: int):
    try:
        # 1) LLM (락 밖)
        try:
            prompt = build_llm_prompt(
                user_prompt=battle_job["user_prompt"],
                memory_text="",
                lang="ko",
                context=battle_job["context"],
                pairs_summary=battle_job["pairs_summary"],
            )
            msg = await asyncio.wait_for(
                get_llm().ainvoke([HumanMessage(content=prompt)]),
                timeout=20,
            )
            assistant_text = msg.content

        except asyncio.CancelledError:
            # ✅ cancel되면 조용히 종료 (cleanup은 finally에서)
            return

        except asyncio.TimeoutError:
            assistant_text = "(assistant) 배틀 해설 생성 시간 초과."
        except Exception as e:
            assistant_text = f"(assistant) 배틀 해설 생성 실패: {e}"

        # 2) 결과 반영 (락 안)
        async with room.lock:
            # ✅ 라운드가 이미 바뀌었으면(상대 나감/리셋/다음라운드) 무시
            if nonce != room.battle_nonce:
                return

            if len(room.players) < 2:
                room.phase = "pick"
                room.winner_player_id = None
                room.chat.append({
                    "role": "system",
                    "sender": "system",
                    "content": "상대가 나가서 배틀이 취소되었습니다.",
                    "ts": time.time(),
                })
                final_payload = {"type": "room_update", "room": snapshot_room(room)}
            else:
                room.chat.append({
                    "role": "assistant",
                    "sender": "assistant",
                    "content": assistant_text,
                    "ts": time.time(),
                })

                loser_id = battle_job.get("loser_player_id")
                if loser_id:
                    loser = room.find_player(loser_id)
                    if loser:
                        loser.lives = max(0, loser.lives - 1)

                p1, p2 = room.players[0], room.players[1]
                match_over = (p1.lives <= 0) or (p2.lives <= 0)

                if match_over:
                    room.winner_player_id = battle_job["winner_player_id"]
                    room.chat.append({
                        "role": "system",
                        "sender": "system",
                        "content": f"게임 종료! 승자: {battle_job['win_name']}",
                        "ts": time.time(),
                    })
                    room.phase = "ended"
                else:
                    for p in room.players:
                        p.picked = False
                        p.pokemon_name = None
                        p.pokemon_doc = None

                    if loser_id:
                        loser_idx = 0 if room.players[0].player_id == loser_id else 1
                        room.turn_index = loser_idx

                    room.winner_player_id = None
                    room.phase = "pick"
                    room.chat.append({
                        "role": "system",
                        "sender": "system",
                        "content": "라운드 종료! 다음 라운드 포켓몬 다시 선택!",
                        "ts": time.time(),
                    })

                final_payload = {"type": "room_update", "room": snapshot_room(room)}

        await broadcast_room(room, final_payload)
        asyncio.create_task(run_bot_until_human_turn(room))

    finally:
        # ✅ 어떤 이유로 끝나든 battle_task 정리 (cancel/예외 포함)
        async with room.lock:
            # 여기가 현재 task인지 확인하고 지우는게 더 안전
            cur = asyncio.current_task()
            if room.battle_task is cur:
                room.battle_task = None



def reset_room(room: RoomState):
    room.phase = "pick"
    room.winner_player_id = None
    room.turn_index = 0
    room.chat = []

    # battle/task 관련
    room.battle_nonce += 1
    t = room.battle_task
    room.battle_task = None
    if t:
        try:
            t.cancel()
        except Exception:
            pass
    # 플레이어 상태 초기화(남아있는 플레이어 기준)
    for p in room.players:
        p.picked = False
        p.pokemon_name = None
        p.pokemon_doc = None
        p.lives = 3


@app.get("/rooms")
async def get_rooms():
    data = []
    for r in rooms.values():
        has_bot = room_has_bot(r)  
        in_progress = room_in_progress(r)
        data.append({
            "room_id": r.room_id,
            "count": len(r.players),
            "is_full": r.is_full(),
            "turn_player_id": r.current_player_id(),
            "has_bot": has_bot,          # ✅ 추가
            "single_play": bool(r.auto_play and has_bot),  # ✅ 추가(참고용)
            "in_progress": in_progress,  # ✅ 추가
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
        if room_has_bot(room):
            raise HTTPException(409, "single play is running")
        if room.find_player(player_id):
            return {"room": snapshot_room(room)}
        if room.is_full():
            raise HTTPException(409, "room is full")

        # 방이 비어있던 상태면 새 게임 초기화
        if len(room.players) == 0:
            reset_room(room)

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
        if room.auto_play:
            bot = next((p for p in room.players if is_bot(p.player_id)), None)
            if bot:
                room.players = [p for p in room.players if p.player_id != bot.player_id]

            room.used_bot_names = set()
            room.auto_play = False
            reset_room(room)

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
            await websocket.send_text(json.dumps(
                {"type": "notice", "code": "NEED_JOIN", "message": "방에 다시 입장해주시기 바랍니다."},
                ensure_ascii=False
            ))
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

            msg_type = data.get("type")
            payload = None

            async with room.lock:
                if len(room.players) < 2:
                    await websocket.send_text(json.dumps(
                        {"type": "notice", "code": "WAITING", "message": "상대방 입장 대기 중"},
                        ensure_ascii=False
                    ))
                    continue

                # 배틀 계산 중이면 입력 막기
                if room.phase == "battle_running":
                    await websocket.send_text(json.dumps(
                        {"type": "notice", "code": "BATTLE_RUNNING", "message": "배틀 계산 중... 잠깐만"},
                        ensure_ascii=False
                    ))
                    continue

                if room.phase == "ended":
                    await websocket.send_text(json.dumps(
                        {"type": "notice", "code": "ENDED", "message": "게임 끝남. 방을 나가거나 새로 시작해."},
                        ensure_ascii=False
                    ))
                    continue

                # 턴 체크
                if room.current_player_id() != player_id:
                    await websocket.send_text(json.dumps(
                        {"type": "notice", "code": "NOT_YOUR_TURN", "message": "상대의 턴입니다."},
                        ensure_ascii=False
                    ))
                    continue

                player = room.find_player(player_id)
                if not player:
                    await websocket.send_text(json.dumps(
                        {"type": "notice", "code": "NOT_IN_ROOM", "message": "방에 없음"},
                        ensure_ascii=False
                    ))
                    continue

                # ===== PICK 단계 =====
                if room.phase == "pick":
                    if msg_type != "pick":
                        await websocket.send_text(json.dumps(
                            {"type": "notice", "code": "NEED_PICK", "message": "첫 턴엔 포켓몬 이름을 선택해야 함"},
                            ensure_ascii=False
                        ))
                        continue

                    name = (data.get("name") or "").strip()
                    if not name:
                        await websocket.send_text(json.dumps(
                            {"type": "notice", "code": "EMPTY", "message": "포켓몬 이름 비었음"},
                            ensure_ascii=False
                        ))
                        continue

                    hit = find_pokemon_by_name(name)
                    if not hit:
                        await websocket.send_text(json.dumps(
                            {"type": "notice", "code": "NOT_FOUND", "message": "데이터에서 포켓몬 못찾음. 다른 이름으로 다시."},
                            ensure_ascii=False
                        ))
                        continue

                    _, doc = hit
                    player.picked = True
                    player.pokemon_name = name
                    player.pokemon_doc = doc

                    room.chat.append({
                        "role": "system",
                        "sender": "system",
                        "content": f"{player.nickname} 선택 완료!",
                        "ts": time.time()
                    })

                    room.turn_index = (room.turn_index + 1) % len(room.players)

                    # ✅ 둘 다 picked면 여기서 battle_running + job 생성 + 스케줄
                    battle_job = make_battle_job(room)
                    if battle_job:
                        schedule_battle(room, battle_job)

                    payload = {"type": "room_update", "room": snapshot_room(room)}

                else:
                    # pick 외 상태는 여기서 다 막고 있어서 사실상 안 옴
                    payload = {"type": "room_update", "room": snapshot_room(room)}

            # lock 밖: 방송
            if payload:
                await broadcast_room(room, payload)
                asyncio.create_task(run_bot_until_human_turn(room))

    except WebSocketDisconnect:
        async with room.lock:
            await _remove_player_from_room(room, player_id, reason="연결 끊김")
            payload = {"type": "room_update", "room": snapshot_room(room)}
        await broadcast_room(room, payload)


@app.post("/rooms/{room_id}/single_play")
async def enable_single_play(room_id: int, player_id: str):
    room = rooms.get(room_id)
    if not room:
        raise HTTPException(404, "room not found")

    async with room.lock:
        me = room.find_player(player_id)
        if not me:
            raise HTTPException(400, "not in room")

        # 이미 2명이면 싱글플레이 못 켬
        if len(room.players) >= 2:
            raise HTTPException(409, "already has opponent")

        reset_room(room)
        # 이미 봇이 있으면 그냥 켜진 상태로 간주
        bot = next((p for p in room.players if is_bot(p.player_id)), None)
        if bot is None:
            bot_id = f"{BOT_PREFIX}{room_id}"
            bot = Player(bot_id, "AI")
            bot.auto_play = True
            room.players.append(bot)
        room.used_bot_names = set()
        room.auto_play = True
        me.auto_play = False
        room.players.sort(key=lambda p: 0 if p.player_id == me.player_id else 1)
        room.turn_index = 0
        room.chat.append({
            "role": "system",
            "sender": "system",
            "content": "싱글플레이 시작!",
            "ts": time.time(),
        })

        # 선턴은 기존 룰 유지(원하면 여기서 무조건 me로 고정해도 됨)
        if len(room.players) == 2 and room.phase == "pick":
            # 기존 join에서 2명 되면 게임 시작 로그 남기듯이 맞춰주기
            room.turn_index = 0

        payload = {"type": "room_update", "room": snapshot_room(room)}

    await broadcast_room(room, payload)

    # ✅ 봇 턴이면 즉시 한 번 굴려주기
    asyncio.create_task(run_bot_until_human_turn(room))
    return {"room": payload["room"]}
