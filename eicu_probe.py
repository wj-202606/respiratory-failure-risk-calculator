import pandas as pd
import os

PATH = r"D:\eicu数据库\EICU 2.0数据"

print("=" * 60)
print("文件列表（前 20 个）")
files = [f for f in os.listdir(PATH) if f.endswith((".csv", ".csv.gz"))]
for f in files[:20]:
    print("  ", f)
print(f"总文件数: {len(files)}")

print("=" * 60)
print("patient.csv.gz")
pat = pd.read_csv(os.path.join(PATH, "patient.csv.gz"), nrows=5)
print("columns:", pat.columns.tolist())
print(pat.head())

print("=" * 60)
print("lab.csv.gz (只读 labname 列，去重)")
lab = pd.read_csv(os.path.join(PATH, "lab.csv.gz"), usecols=["labname"])
names = lab["labname"].dropna().unique()
print(f"labname 总种类: {len(names)}")
for kw in ["alb", "bun", "urea", "base", "bicarb", "hco3"]:
    hits = [n for n in names if kw in str(n).lower()]
    print(f"\n包含 '{kw}' 的 labname ({len(hits)} 个):")
    for h in hits[:20]:
        print("  ", h)