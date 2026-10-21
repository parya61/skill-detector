import json
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

EMB = np.load("embeddings_e5small.npy")
with open("emb_ids_e5small.json", encoding="utf-8") as f:
    ids = json.load(f)
idx = {bid: i for i, bid in enumerate(ids)}      # benchmark_id → строка в матрице

prim = load_dataset("ProtectSkills/MaliciousSkillBench", "primary")["train"]
by_id = {r["benchmark_id"]: int(r["label"]) for r in prim}

results = {"model": "e5_small__logreg"}

for proto in ["random", "source_balanced_random", "m_structural_disjoint", "source_disjoint"]:
    sp = load_dataset("ProtectSkills/MaliciousSkillBench", "splits", split=proto)
    tr = [r["benchmark_id"] for r in sp if r["split"] == "train"]
    te = [r["benchmark_id"] for r in sp if r["split"] in ("test", "validation")]

    Xtr = EMB[[idx[i] for i in tr]]; ytr = [by_id[i] for i in tr]
    Xte = EMB[[idx[i] for i in te]]; yte = [by_id[i] for i in te]

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

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

with open("emb_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\nсохранил → emb_results.json")
