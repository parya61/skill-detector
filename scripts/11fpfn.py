# Разбор FP/FN на source_disjoint: кого детектор валит зря и кого пропускает
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from collections import Counter

prim = load_dataset("ProtectSkills/MaliciousSkillBench", "primary")["train"]
by_id = {r["benchmark_id"]: (int(r["label"]), r["skill_text"] or "") for r in prim}
src = {r["benchmark_id"]: r.get("source_name") or r.get("source_id") for r in prim}

sp = load_dataset("ProtectSkills/MaliciousSkillBench", "splits", split="source_disjoint")
tr = [r["benchmark_id"] for r in sp if r["split"] == "train"]
te = [r["benchmark_id"] for r in sp if r["split"] in ("test", "validation")]
Xtr = [by_id[i][1] for i in tr]; ytr = [by_id[i][0] for i in tr]
Xte = [by_id[i][1] for i in te]; yte = [by_id[i][0] for i in te]

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)),
    ("clf", LinearSVC(class_weight="balanced", max_iter=2000)),
])
pipe.fit(Xtr, ytr)
pred = pipe.predict(Xte)

# label: 0 = безопасный, 1 = вредоносный
fp = [i for i in range(len(yte)) if yte[i] == 0 and pred[i] == 1]  # безопасные, заваленные
fn = [i for i in range(len(yte)) if yte[i] == 1 and pred[i] == 0]  # вредоносные, пропущенные

print(f"тест: {len(yte)} | FP (безопасный→вредоносный): {len(fp)} | FN (вредоносный→безопасный): {len(fn)}")

print("\n=== FP по источникам (кого валит зря) ===")
for s, c in Counter(src[te[i]] for i in fp).most_common(10):
    print(f"  {c:4d}  {s}")

print("\n=== примеры FP (безопасные, которых завалили) ===")
for i in fp[:5]:
    print(f"  [{src[te[i]]}] {by_id[te[i]][1][:180]}")

print("\n=== примеры FN (вредоносные, которых пропустили) ===")
for i in fn[:5]:
    print(f"  [{src[te[i]]}] {by_id[te[i]][1][:180]}")
