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

def enable_single_play(room_id, player_id):
    return requests.post(
        f"{API}/rooms/{room_id}/single_play",
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
if "ended_ts" not in st.session_state:
    st.session_state.ended_ts = None
if "auto_leaving" not in st.session_state:
    st.session_state.auto_leaving = False
if "joining" not in st.session_state:
    st.session_state.joining = False

drain_ws_messages()

st.title("포켓몬 배틀 시뮬레이션 (방/웹소켓 + 서버 AI)")


# =========================
# lobby
# =========================
if st.session_state.room_id is None:
    lobby_lock = st.session_state.joining
    st.subheader("방 목록")
    st.session_state.nickname = st.text_input("닉네임", st.session_state.nickname, disabled=lobby_lock)
    if "pending_join_room_id" not in st.session_state:
        st.session_state.pending_join_room_id = None
    rooms = get_rooms()
    if lobby_lock:
        st.info("입장중입니다. 잠시만 기다려주세요.")
    if st.session_state.pending_join_room_id is not None:
        rid = st.session_state.pending_join_room_id
        try:
            try:
                res = join_room(rid, st.session_state.player_id, st.session_state.nickname)
            except Exception as e:
                st.session_state.joining = False
                st.session_state.pending_join_room_id = None
                st.error(f"입장 요청 실패: {e}")
                st.stop()
            if "room" not in res:
                st.session_state.joining = False
                st.session_state.pending_join_room_id = None
                st.error(f"입장 실패: {res}")
                st.stop()

            ws_close()
            st.session_state.room_id = rid
            st.session_state.room = res["room"]
            st.session_state.last_notice = None
            st.session_state.ws_thread_alive = False
            st.session_state.ws_connected = False

        finally:
            # ✅ 성공/실패 상관없이 pending 정리
            st.session_state.pending_join_room_id = None
            st.session_state.joining = False

        st.rerun()

    for r in rooms:
        cols = st.columns([1, 1, 2, 2])
        cols[0].write(f"방 {r['room_id']}")
        cols[1].write(f"{r['count']}/2")
        status = []
        if r.get("has_bot"):
            status.append("🤖 싱글플레이 진행중")
        elif r.get("in_progress") and r["count"] >= 2:
            status.append("🟠 진행중")
        else:
            status.append("🟢 대기중")
        cols[2].write(" / ".join(status))

        # ✅ 입장 버튼 disable 규칙
        disabled = st.session_state.joining or r["is_full"] or r.get("has_bot", False)
        if cols[3].button("입장", disabled=disabled, key=f"join{r['room_id']}"):
            st.session_state.pending_join_room_id = r["room_id"]
            st.session_state.joining = True
            st.rerun()
            

# =========================
# room
# =========================
else:
    room_id = st.session_state.room_id
    my_id = st.session_state.player_id

    ensure_ws_started(room_id, my_id)
    changed = drain_ws_messages()

    room = st.session_state.room
    if changed:
        st.rerun()
    msg = st.session_state.get("last_notice")
    ts = st.session_state.get("notice_ts")

    if msg and ts and st.session_state.get("last_notice_shown_ts") != ts:
        st.toast(msg)
        st.session_state.last_notice_shown_ts = ts
    st.subheader(f"방 {room_id}")
    st.caption("🟢 WS: connected" if st.session_state.ws_connected else "🟡 WS: connecting...")
    if not st.session_state.ws_connected:
        st.markdown("""
        <style>
        .room-lock {
            position: fixed;
            inset: 0;
            background: rgba(255,255,255,0.35);
            backdrop-filter: blur(4px);
            z-index: 9990;
            pointer-events: none;
        }
        .room-lock-msg{
            position: fixed;
            top: 64px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 9991;
            background: rgba(0,0,0,0.75);
            color: white;
            padding: 10px 14px;
            border-radius: 10px;
            font-size: 14px;
            pointer-events: none;
        }
        </style>
        <div class="room-lock"></div>
        <div class="room-lock-msg">웹소켓 연결중...</div>
        """, unsafe_allow_html=True)
    
    if st.button("나가기"):
        try:
            leave_room(room_id, my_id)
        except Exception:
            pass
        ws_close()
        st.session_state.room_id = None
        st.session_state.room = None
        st.rerun()

    if room and room.get("auto_play"):
        st.caption("🤖 싱글플레이: AI 상대 활성화중")
    if not st.session_state.ws_connected:
        st.warning("웹소켓 연결중입니다. 잠시만 기다려주세요.")
        time.sleep(0.3)
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

    # ✅ ended 진입/이탈 감지
    if phase == "ended":
        if st.session_state.ended_ts is None:
            st.session_state.ended_ts = time.time()
            st.session_state.auto_leaving = False
    else:
        st.session_state.ended_ts = None
        st.session_state.auto_leaving = False

    # ✅ ended 상태면: 15초 타이머 후 자동 퇴장
    if phase == "ended" and st.session_state.ended_ts is not None:
        elapsed = time.time() - st.session_state.ended_ts
        remain = max(0, 15 - int(elapsed))
        st.info(f"15초 후 자동으로 대기실로 이동합니다. (남은 시간: {remain}초)")

        if elapsed >= 15 and not st.session_state.auto_leaving:
            st.session_state.auto_leaving = True
            try:
                leave_room(room_id, my_id)
            except Exception:
                pass
            ws_close()
            st.session_state.room_id = None
            st.session_state.room = None
            st.session_state.ended_ts = None
            st.session_state.auto_leaving = False
            st.rerun()

    if len(players) < 2:
        st.warning("상대방 입장 대기 중 (2명 되면 시작)")
        auto_play = bool(room.get("auto_play"))
        if not auto_play:
            if st.button("싱글플레이 시작 (AI 상대)"):
                res = enable_single_play(room_id, my_id)
                # 서버가 room_update를 브로드캐스트하지만, 즉시 반영하려고 로컬도 업데이트
                if "room" in res:
                    st.session_state.room = res["room"]
                st.rerun()
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

    
    time.sleep(0.2)
    st.rerun()
