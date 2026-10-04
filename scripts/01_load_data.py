from datasets import load_dataset

# качаем бенчмарк с Hugging Face; "primary" — основная таблица
ds = load_dataset("ProtectSkills/MaliciousSkillBench", "primary")
print(ds)

d = ds["train"]
print("строк:", len(d))
print("колонки:", d.column_names)

row = d[0]
for k, v in row.items():
    print(f"  {k}: {str(v)[:120]}")
