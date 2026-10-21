import json
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"   # лёгкая: 118M, 384-dim

prim = load_dataset("ProtectSkills/MaliciousSkillBench", "primary")["train"]
ids = [r["benchmark_id"] for r in prim]
texts = ["passage: " + (r["skill_text"] or "") for r in prim]  # e5 требует префикс "passage: "

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

model = SentenceTransformer(MODEL_NAME, device=device)
model.max_seq_length = 512          # ← обрезаем гигантские тексты

emb = model.encode(
    texts,
    batch_size=32,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)
print("shape:", emb.shape)          # ждём (9740, 384)

np.save("embeddings_e5small.npy", emb)
with open("emb_ids_e5small.json", "w", encoding="utf-8") as f:
    json.dump(ids, f)
print("сохранил → embeddings_e5small.npy + emb_ids_e5small.json")
