from collections import defaultdict
from datasets import load_dataset

splits = load_dataset("ProtectSkills/MaliciousSkillBench", "splits")

for name in splits:
    s = splits[name]
    parts = defaultdict(int)     # сколько примеров в train/validation/test
    srcs = defaultdict(set)      # какие ИСТОЧНИКИ в каждом куске
    for row in s:
        parts[row["split"]] += 1
        srcs[row["split"]].add(row["source_id"])
    print(f"\n=== {name} ===")
    print("  примеров:", dict(parts))
    print("  источников:", {k: len(v) for k, v in srcs.items()})
