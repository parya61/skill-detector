from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

prim = load_dataset("ProtectSkills/MaliciousSkillBench", "primary")["train"]
by_id = {r["benchmark_id"]: (int(r["label"]), r["skill_text"] or "") for r in prim}

for proto in ["random", "source_balanced_random", "m_structural_disjoint", "source_disjoint"]:
    sp = load_dataset("ProtectSkills/MaliciousSkillBench", "splits", split=proto)
    tr = [r["benchmark_id"] for r in sp if r["split"] == "train"]
    te = [r["benchmark_id"] for r in sp if r["split"] in ("test", "validation")]
    Xtr = [by_id[i][1] for i in tr]; ytr = [by_id[i][0] for i in tr]
    Xte = [by_id[i][1] for i in te]; yte = [by_id[i][0] for i in te]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)),
    ])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    print(f"\n=== {proto} ===")
    print("accuracy:", round(accuracy_score(yte, pred), 3),
          "| macro-F1:", round(f1_score(yte, pred, average="macro"), 3))
    print(classification_report(yte, pred, target_names=["безопасный", "вредоносный"], digits=3))
