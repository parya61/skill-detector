from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

import json

prim = load_dataset("ProtectSkills/MaliciousSkillBench", "primary")["train"]
by_id = {r["benchmark_id"]: (int(r["label"]), r["skill_text"] or "") for r in prim}

results = {"model": "char_wb_3_5"}

for proto in ["random", "source_balanced_random", "m_structural_disjoint", "source_disjoint"]:
    sp = load_dataset("ProtectSkills/MaliciousSkillBench", "splits", split=proto)
    tr = [r["benchmark_id"] for r in sp if r["split"] == "train"]
    te = [r["benchmark_id"] for r in sp if r["split"] in ("test", "validation")]
    Xtr = [by_id[i][1] for i in tr]; ytr = [by_id[i][0] for i in tr]
    Xte = [by_id[i][1] for i in te]; yte = [by_id[i][0] for i in te]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5),
                          min_df=3, sublinear_tf=True, max_features=200000)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)),
    ])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    print(f"\n=== {proto} ===")
    print("accuracy:", round(accuracy_score(yte, pred), 3),
          "| macro-F1:", round(f1_score(yte, pred, average="macro"), 3))
    
    print(classification_report(yte, pred, target_names=["безопасный", "вредоносный"], digits=3))

    rep = classification_report(yte, pred, target_names=["безопасный", "вредоносный"],
                            digits=3, output_dict=True)
    results[proto] = {
    "acc": rep["accuracy"],
    "macro_f1": rep["macro avg"]["f1-score"],
    "benign_recall": rep["безопасный"]["recall"],
    "benign_prec": rep["безопасный"]["precision"],
    "mal_recall": rep["вредоносный"]["recall"],
    "mal_prec": rep["вредоносный"]["precision"],
    }
with open("char_wb_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\nсохранил → char_wb_results.json")
