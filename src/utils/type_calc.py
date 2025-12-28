import json
from pathlib import Path

def load_type_chart():
    path = Path("src/data/type_chart.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

TYPE_CHART = load_type_chart()

def calc_multiplier(atk_type: str, def_types: list[str]) -> float:
    atk = atk_type.lower()
    mult = 1.0
    for d in def_types:
        if not d or str(d).lower() == "nan":
            continue
        mult *= TYPE_CHART.get(atk, {}).get(d.lower(), 1.0)
    return mult
