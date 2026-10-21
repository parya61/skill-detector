import time
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

prim = load_dataset("ProtectSkills/MaliciousSkillBench", "primary")["train"]
texts = [r["skill_text"] or "" for r in prim]
lens = np.array([len(t) for t in texts])
print("n:", len(texts), "| chars min/med/p90/max:",
      lens.min(), int(np.median(lens)), int(np.percentile(lens, 90)), lens.max())

model = SentenceTransformer("BAAI/bge-m3", device="cuda")
model.max_seq_length = 512     # ← главный рычаг: режем контекст
model.half()                   # ← fp16: вдвое меньше памяти, для Turing быстрее

sample = texts[:64]
t0 = time.time()
model.encode(sample, batch_size=8, normalize_embeddings=True)
dt = time.time() - t0
rate = len(sample) / dt
print(f"{len(sample)} текстов за {dt:.1f}s = {rate:.1f} docs/s → ETA всех: {len(texts)/rate/60:.1f} мин")
