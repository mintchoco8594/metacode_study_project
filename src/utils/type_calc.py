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
