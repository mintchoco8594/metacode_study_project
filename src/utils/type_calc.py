# utils/type_calc.py
import json
import re
from pathlib import Path
import numpy as np


def label_by_multiplier(mult: float) -> str:
    if mult == 0:
        return "무효(0배)"
    if mult >= 4:
        return "매우 유리(4배)"
    if mult >= 2:
        return "유리(2배)"
    if mult == 1:
        return "보통(1배)"
    if mult <= 0.25:
        return "매우 불리(0.25배)"
    if mult <= 0.5:
        return "불리(0.5배)"
    return "애매"
    
def load_type_chart(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))

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


def calc_best_stab_attack(attacker_doc: dict, defender_doc: dict, chart: dict) -> dict:
    atk_types = attacker_doc.get("meta", {}).get("types", []) or []
    def_types = defender_doc.get("meta", {}).get("types", []) or []

    # 타입 데이터 없으면 fallback
    if not atk_types:
        move_type = "normal"
        type_mult = calc_type_multiplier(move_type, def_types, chart)
        stab = 1.0
        final_mult = type_mult * stab
        return {
            "move_type": move_type,
            "type_mult": type_mult,
            "stab": stab,
            "mult": final_mult,
            "label": label_by_multiplier(type_mult),
        }

    # 공격자 타입 중 상대에게 가장 유리한 타입 선택
    best_type = None
    best_type_mult = -1.0
    for ty in atk_types:
        m = calc_type_multiplier(ty, def_types, chart)
        if m > best_type_mult:
            best_type_mult = m
            best_type = ty

    stab = 1.5
    final_mult = best_type_mult * stab
    return {
        "move_type": best_type,
        "type_mult": best_type_mult,
        "stab": stab,
        "mult": final_mult,
        "label": label_by_multiplier(best_type_mult),
    }

def calc_given_type_attack(attacker_doc: dict, defender_doc: dict, move_type: str, chart: dict) -> dict:
    def_types = defender_doc.get("meta", {}).get("types", []) or []
    atk_types = attacker_doc.get("meta", {}).get("types", []) or []

    move_type = (move_type or "normal").strip().lower()
    type_mult = calc_type_multiplier(move_type, def_types, chart)
    stab = 1.5 if move_type in [t.lower() for t in atk_types] else 1.0
    final_mult = type_mult * stab

    return {
        "move_type": move_type,
        "type_mult": type_mult,
        "stab": stab,
        "mult": final_mult,
        "label": label_by_multiplier(type_mult),
    }


TYPE_Q_WORDS = {"약점", "강점", "저항", "상성", "효과", "반감", "배율", "상대로", "유리", "불리", "뭐야", "알려줘", "설명"}

def is_type_question(text: str, valid_types: set[str]) -> bool:
    s = (text or "").strip().lower()
    if not s:
        return False

    # 1) "타입" 단어가 있거나
    if "타입" in s:
        return True

    # 2) 타입명 + 상성 키워드 조합
    has_type = any(re.search(rf"\b{re.escape(t)}\b", s, flags=re.IGNORECASE) for t in valid_types)
    has_qword = any(w in s for w in TYPE_Q_WORDS)
    return has_type and has_qword


def build_type_context(chart: dict, valid_types: set[str], user_text: str, lang: str = "ko") -> str:
    """
    - 유저가 특정 타입을 물으면 그 타입 기준(공격/방어) 핵심만 뽑아서 근거로 줌
    - 특정 타입을 못 집으면 전체 타입차트 요약(짧게) 근거로 줌
    """
    s = (user_text or "").strip()
    low = s.lower()

    # 유저가 언급한 "대상 타입(방어 타입)" 또는 "기술 타입(공격 타입)"을 하나라도 잡아보자
    mentioned = []
    for t in valid_types:
        if re.search(rf"\b{re.escape(t)}\b", low, flags=re.IGNORECASE):
            mentioned.append(t)


    for ko, en in TYPE_ALIAS_KO.items():
        if ko in s and en in valid_types and en not in mentioned:
            mentioned.append(en)

    def fmt_line(title, items):
        if not items:
            return f"{title}: 없음"
        return f"{title}: " + ", ".join(items)

    def analyze_defense(def_type: str):
        # 어떤 공격 타입이 이 방어 타입에 2배/0.5배/0배냐
        weak, resist, immune = [], [], []
        for atk in valid_types:
            mult = chart[atk].get(def_type, 1.0)  # chart: 공격타입 -> 방어타입 -> 배율 (네 코드 기준)
            if mult >= 2.0:
                weak.append(atk)
            elif 0.0 < mult <= 0.5:
                resist.append(atk)
            elif mult == 0.0:
                immune.append(atk)
        return weak, resist, immune

    def analyze_attack(atk_type: str):
        # 이 공격 타입이 어떤 방어 타입에 2배/0.5배/0배냐
        strong, weak, no = [], [], []
        row = chart.get(atk_type, {})
        for def_type in valid_types:
            mult = row.get(def_type, 1.0)
            if mult >= 2.0:
                strong.append(def_type)
            elif 0.0 < mult <= 0.5:
                weak.append(def_type)
            elif mult == 0.0:
                no.append(def_type)
        return strong, weak, no

    # 1개라도 타입을 잡았으면 그 타입 기준 근거 생성 (너무 길어지지 않게 1~2개만)
    if mentioned:
        mentioned = mentioned[:2]
        blocks = []
        for t in mentioned:
            w, r, im = analyze_defense(t)
            stg, wk, no = analyze_attack(t)

            if lang == "en":
                blocks.append(
                    "\n".join([
                        "[EVIDENCE: Type Chart]",
                        f"Type: {t}",
                        fmt_line("Defense - weak to (2x)", w),
                        fmt_line("Defense - resists (0.5x)", r),
                        fmt_line("Defense - immune (0x)", im),
                        fmt_line("Attack - strong against (2x)", stg),
                        fmt_line("Attack - not effective (0.5x)", wk),
                        fmt_line("Attack - no effect (0x)", no),
                    ])
                )
            else:
                blocks.append(
                    "\n".join([
                        "[타입 상성 근거]",
                        f"타입: {t}",
                        fmt_line("방어 기준 약점(2배)", w),
                        fmt_line("방어 기준 반감(0.5배)", r),
                        fmt_line("방어 기준 무효(0배)", im),
                        fmt_line("공격 기준 유리(2배)", stg),
                        fmt_line("공격 기준 반감(0.5배)", wk),
                        fmt_line("공격 기준 무효(0배)", no),
                    ])
                )
        return "\n\n".join(blocks).strip()

    # 타입을 못 잡으면 “타입차트가 존재한다” 정도만 짧게 안내용 근거
    if lang == "en":
        return "[EVIDENCE: Type Chart]\nType effectiveness chart is available, but the requested type was not detected from the question."
    return "[타입 상성 근거]\n타입 상성표(type_chart.json)는 로드되어 있으나, 질문에서 특정 타입을 식별하지 못했어."
