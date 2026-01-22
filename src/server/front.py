# front.py
import json
import uuid
import queue
import threading
import time

import requests
import streamlit as st
import websocket  # websocket-client

import logging

API = "http://localhost:8000"
WS_BASE = "ws://localhost:8000"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger(__name__)
logging.getLogger("websocket").setLevel(logging.WARNING)

st.set_page_config(page_title="Pokemon Battle Rooms (WS)", layout="wide")


def get_ws_holder():
    if "ws_holder" not in st.session_state:
        st.session_state.ws_holder = {"ws": None}
    return st.session_state.ws_holder
# =========================
# REST helpers
# =========================
def get_rooms():
    return requests.get(f"{API}/rooms", timeout=30).json()["rooms"]

def join_room(room_id, player_id, nickname):
    return requests.post(
        f"{API}/rooms/{room_id}/join",
        params={"player_id": player_id, "nickname": nickname},
        timeout=10,
    ).json()

def leave_room(room_id, player_id):
    return requests.post(
        f"{API}/rooms/{room_id}/leave",
        params={"player_id": player_id},
        timeout=10,
    ).json()


# =========================
# WS helpers
# =========================
def ws_url(room_id: int, player_id: str) -> str:
    return f"{WS_BASE}/ws/rooms/{room_id}/{player_id}"

def ensure_ws_started(room_id: int, player_id: str):
    holder = get_ws_holder()

    t = st.session_state.get("ws_thread")
    if t is not None and t.is_alive() and holder["ws"] is not None:
        return

    if "ws_queue" not in st.session_state:
        st.session_state.ws_queue = queue.Queue()
    q = st.session_state.ws_queue

    def on_message(ws, message: str):
        q.put(message)

    def on_error(ws, error):
        q.put(json.dumps({"type": "ws_error", "message": str(error)}, ensure_ascii=False))

    def on_open(ws):
        q.put(json.dumps({"type": "ws_open"}, ensure_ascii=False))

    def on_close(ws, close_status_code, close_msg):
        q.put(json.dumps({"type": "ws_closed", "code": close_status_code, "msg": close_msg}, ensure_ascii=False))
        holder["ws"] = None

    def run():
        ws = websocket.WebSocketApp(
            ws_url(room_id, player_id),
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        holder["ws"] = ws
        ws.run_forever(ping_interval=20, ping_timeout=10)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    st.session_state.ws_thread = th
    st.session_state.ws_thread_alive = True

def ws_send_pick(name: str) -> bool: 
    ws = get_ws_holder()["ws"] 
    if not ws: 
        logger.debug("ws") 
        return False 
    try: 
        logger.debug("전송성공") 
        ws.send(json.dumps({"type": "pick", "name": name}, ensure_ascii=False)) 
        return True 
    except Exception: 
        logger.debug("예외발생") 
        return False

def ws_send_chat(content: str) -> bool:
    ws = get_ws_holder()["ws"]
    if not ws:
        return False
    try:
        ws.send(json.dumps({"type": "chat", "content": content}, ensure_ascii=False))
        return True
    except Exception:
        return False


def ws_close():
    holder = get_ws_holder()
    ws = holder["ws"]
    if ws:
        try:
            ws.close()
        except Exception:
            pass
    holder["ws"] = None
    st.session_state.ws_thread_alive = False
    st.session_state.ws_connected = False




# =========================
# WS message drain
# =========================
def drain_ws_messages():
    if "ws_queue" not in st.session_state:
        return False

    changed = False
    while True:
        try:
            raw = st.session_state.ws_queue.get_nowait()
        except queue.Empty:
            break

        try:
            data = json.loads(raw)
        except Exception:
            continue

        t = data.get("type")
        if t in ("room_snapshot", "room_update"):
            st.session_state.room = data.get("room")
            changed = True
        elif t == "notice":
            st.session_state.last_notice = data.get("message")
            st.session_state.notice_ts = time.time()   # 추가
            changed = True
        elif t == "ws_error":
            st.session_state.last_notice = f"WS 에러: {data.get('message')}"
            changed = True
        elif t == "ws_open":
            st.session_state.ws_connected = True
            st.session_state.last_notice = None
            st.session_state.notice_ts = None
            changed = True
        elif t == "ws_closed":
            st.session_state.ws_connected = False
            st.session_state.ws_thread_alive = False
            st.session_state.last_notice = "웹소켓 끊김"
            st.session_state.notice_ts = time.time()
            changed = True

    return changed



# =========================
# state init
# =========================
if "player_id" not in st.session_state:
    st.session_state.player_id = str(uuid.uuid4())
if "nickname" not in st.session_state:
    st.session_state.nickname = "player"
if "room_id" not in st.session_state:
    st.session_state.room_id = None
if "room" not in st.session_state:
    st.session_state.room = None
if "ws_thread_alive" not in st.session_state:
    st.session_state.ws_thread_alive = False
if "ws_connected" not in st.session_state:
    st.session_state.ws_connected = False
if "last_notice" not in st.session_state:
    st.session_state.last_notice = None
if "ws_queue" not in st.session_state:
    st.session_state.ws_queue = queue.Queue()
if "notice_ts" not in st.session_state:
    st.session_state.notice_ts = None
if "last_notice_shown_ts" not in st.session_state:
    st.session_state.last_notice_shown_ts = None

drain_ws_messages()

st.title("포켓몬 배틀 시뮬레이션 (방/웹소켓 + 서버 AI)")


# =========================
# lobby
# =========================
if st.session_state.room_id is None:
    st.subheader("방 목록")
    st.session_state.nickname = st.text_input("닉네임", st.session_state.nickname)

    rooms = get_rooms()
    for r in rooms:
        cols = st.columns([1, 1, 2])
        cols[0].write(f"방 {r['room_id']}")
        cols[1].write(f"{r['count']}/2")

        if cols[2].button("입장", disabled=r["is_full"], key=f"join{r['room_id']}"):
            res = join_room(r["room_id"], st.session_state.player_id, st.session_state.nickname)

            if "room" not in res:
                st.error(f"입장 실패: {res}")
                st.stop()

            ws_close()

            st.session_state.room_id = r["room_id"]
            st.session_state.room = res["room"]
            st.session_state.last_notice = None
            st.session_state.ws_thread_alive = False
            st.session_state.ws_connected = False

            st.rerun()


# =========================
# room
# =========================
else:
    room_id = st.session_state.room_id
    my_id = st.session_state.player_id

    ensure_ws_started(room_id, my_id)
    drain_ws_messages()
    room = st.session_state.room

    msg = st.session_state.get("last_notice")
    ts = st.session_state.get("notice_ts")

    if msg and ts and st.session_state.get("last_notice_shown_ts") != ts:
        st.toast(msg)
        st.session_state.last_notice_shown_ts = ts
    st.subheader(f"방 {room_id}")
    st.caption("🟢 WS: connected" if st.session_state.ws_connected else "🟡 WS: connecting...")
    if not st.session_state.ws_connected:
        st.warning("웹소켓 연결중입니다. 잠시만 기다려주세요.")
        time.sleep(0.2)
        st.rerun()
    if st.button("나가기"):
        try:
            leave_room(room_id, my_id)
        except Exception:
            pass
        ws_close()
        st.session_state.room_id = None
        st.session_state.room = None
        st.rerun()

    if not room:
        st.info("방 정보 받는 중...")
        time.sleep(0.2)
        st.rerun()

    players = room.get("players", []) or []

    st.markdown("❤️ 목숨")
    for p in players:
        lives = int(p.get("lives", 0) or 0)
        hearts = "❤️" * lives + "🖤" * (3 - lives)
        st.write(f"- {p.get('nickname')} : {hearts} ({lives}/3)")
    turn_id = room.get("turn_player_id")

    phase = room.get("phase", "pick")


    if len(players) < 2:
        st.warning("상대방 입장 대기 중 (2명 되면 시작)")
    else:
        if turn_id == my_id:
            st.success("나의 턴")
        else:
            st.info("상대 턴")

    for m in room.get("chat", []) or []:
        st.markdown(f"**[{m.get('sender','')}]** {m.get('content','')}")


    if phase == "pick":
        st.info("포켓몬 선택 단계입니다. 사용할 포켓몬 이름을 입력해주세요.")
    elif phase == "battle_running":
        st.info("배틀 진행 중... AI가 배틀 내용을 생성중입니다..")
    elif phase == "ended":
        winner = room.get("winner_player_id")
        if winner == my_id:
            st.success("🎉승리!")
        elif winner:
            st.error("😵 패배...")
        else:
            st.warning("🤝 무승부!")
        st.info("게임 종료! '나가기' 버튼을 눌러주세요.")
    else:
        st.warning(f"알 수 없는 phase: {phase}")

    me = next((p for p in players if p.get("player_id") == my_id), None)
    my_picked = bool(me and me.get("picked"))

    if phase == "pick" and my_picked:
        st.success("✅ 포켓몬 선택 완료! 상대 선택 기다리는 중...")

    disabled = (
        (len(players) < 2)
        or (turn_id is not None and turn_id != my_id)
        or (not st.session_state.ws_connected)
        or (phase in ("battle_running", "ended"))
        or (phase == "pick" and my_picked)
    )

    if phase == "pick":
        with st.form("pick_form", clear_on_submit=True):
            name = st.text_input("포켓몬 이름", key="pick_name", disabled=disabled)
            submitted = st.form_submit_button("선택", disabled=disabled)
        if submitted and name.strip():
            ok = ws_send_pick(name.strip())
            if not ok:
                st.warning("선택 전송 실패(웹소켓 연결 확인)")
            st.rerun()

    if phase != "ended":
        time.sleep(0.2)
        st.rerun()
