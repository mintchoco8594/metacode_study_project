import pandas as pd
from pathlib import Path
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer

CSV_PATH = Path("src/data/pokemon.csv")
OUT_DIR = Path("rag_store")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH, encoding="utf-8")

AGAINST_COLS = [c for c in df.columns if c.startswith("against_")]

def norm_type(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    return "" if s == "nan" else s

def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

def safe_float(x, default=1.0):
    try:
        return float(x)
    except Exception:
        return default

def row_to_text(r):
    num = safe_int(r.get("#"))
    name = str(r.get("english_name", "")).strip()
    
    t1 = norm_type(r.get("primary_type", ""))
    t2 = norm_type(r.get("secondary_type", ""))

    type_str = t1 if not t2 else f"{t1}/{t2}"

    stats = (
        f"Total {safe_int(r.get('Total'))}, "
        f"HP {safe_int(r.get('HP'))}, "
        f"Atk {safe_int(r.get('Attack'))}, "
        f"Def {safe_int(r.get('Defense'))}, "
        f"SpA {safe_int(r.get('Sp. Atk'))}, "
        f"SpD {safe_int(r.get('Sp. Def'))}, "
        f"Spe {safe_int(r.get('Speed'))}"
    )

    against_parts = []
    for c in AGAINST_COLS:
        val = r.get(c)
        if pd.notna(val):
            against_parts.append(f"{c.replace('against_','')}={safe_float(val)}")
    against_str = ", ".join(against_parts)

    desc = str(r.get("description", "")).strip()

    return (
        f"#{num} {name}\n"
        f"Type: {type_str}\n"
        f"Base Stats: {stats}\n"
        f"Damage multipliers received: {against_str}\n"
        f"Description: {desc}"
    )

texts = df.apply(row_to_text, axis=1).tolist()

metas = []
for _, r in df.iterrows():
    t1 = norm_type(r.get("primary_type", ""))
    t2 = norm_type(r.get("secondary_type", ""))
    korean_name = str(r.get("korean_name", "")).strip()
    against_map = {}
    for c in AGAINST_COLS:
        val = r.get(c)
        if pd.notna(val):
            against_map[c.replace("against_", "")] = safe_float(val)

    metas.append({
        "dex_no": safe_int(r.get("#")),
        "english_name": str(r.get("english_name", "")).strip(),
        "korean_name": korean_name,
        "types": [t1, t2] if t2 else [t1],

        "stats": {
            "total": safe_int(r.get("Total")),
            "hp": safe_int(r.get("HP")),
            "atk": safe_int(r.get("Attack")),
            "def": safe_int(r.get("Defense")),
            "spa": safe_int(r.get("Sp. Atk")),
            "spd": safe_int(r.get("Sp. Def")),
            "spe": safe_int(r.get("Speed"))
        },

        "against": against_map
    })

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
emb = np.array(emb, dtype="float32")

dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(emb)

faiss.write_index(index, str(OUT_DIR / "pokemon.index"))
(Path(OUT_DIR / "documents.jsonl")).write_text(
    "\n".join(json.dumps({"text": t, "meta": m}, ensure_ascii=False) for t, m in zip(texts, metas)),
    encoding="utf-8"
)

print("saved:", OUT_DIR / "pokemon.index")
print("saved:", OUT_DIR / "documents.jsonl")
print("count:", index.ntotal)
