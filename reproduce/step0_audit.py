# -*- coding: utf-8 -*-
"""
step0_audit.py
复现第 0 步：环境与数据自检（不做任何假设）
"""
import os, io, sys, csv, glob, re, subprocess

PROJ   = r"C:\Users\24339\Desktop\6.30呼吸衰竭\6.30呼衰\1.R数据预处理+变量筛选9.21"
REPO   = r"C:\Users\24339\Desktop\respiratory-failure-risk-calculator"
FROZEN = os.path.join(PROJ, "results", "FROZEN_v1")

def line(t): print("\n" + "="*70 + "\n" + t + "\n" + "="*70)

# ---------- 1. verify_frozen.py 键数探测 ----------
line("1. verify_frozen.py 探测")
vf = os.path.join(PROJ, "verify_frozen.py")
if not os.path.exists(vf):
    print("[缺失] 未找到 verify_frozen.py:", vf)
else:
    src = io.open(vf, encoding="utf-8", errors="ignore").read()
    keys = re.findall(r'["\'][A-Za-z0-9_\.\-]+["\']\s*:', src)
    print("verify_frozen.py 存在，大小 %d 字节" % len(src))
    print("粗略统计 EXPECTED 键条目数：%d" % len(keys))
    print("→ 接近 96 = v10.0 口径；接近 57 = v9.0 旧口径")
    print("\n尝试运行 verify_frozen.py ...")
    try:
        r = subprocess.run([sys.executable, vf], capture_output=True, text=True, timeout=300)
        print("返回码：", r.returncode)
        print("--- stdout ---"); print(r.stdout[-3000:])
        if r.stderr.strip():
            print("--- stderr ---"); print(r.stderr[-2000:])
    except Exception as e:
        print("运行失败：", repr(e))

# ---------- 2. FROZEN_v1 文件清单 ----------
line("2. FROZEN_v1 文件清单")
if not os.path.isdir(FROZEN):
    print("[缺失] FROZEN_v1 目录不存在:", FROZEN)
else:
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(FROZEN, "*")))
    print("共 %d 个文件：" % len(files))
    for f in files: print("  ", f)
    need = ["external_predictions.csv", "external_predictions_hco3_v2.csv",
            "eicu_predictions_be.csv", "eicu_predictions_hco3.csv",
            "eicu_be_summary.csv", "eicu_hco3_summary.csv",
            "eicu_dca.csv", "dca_results_dual.csv",
            "recal_summary_dual.csv", "recal_bootstrap_summary_v2.csv",
            "recal_bootstrap_eicu_summary.csv",
            "summary_for_manuscript.csv", "auc_100splits.csv"]
    print("\n关键文件存在性：")
    for n in need:
        print("  %-45s %s" % (n, "OK" if os.path.exists(os.path.join(FROZEN, n)) else "缺失"))

# ---------- 3. Figure 2 数据文件与列名 ----------
line("3. Figure 2 所需数据（external_predictions.csv）")
cands = [os.path.join(FROZEN, "external_predictions.csv"),
         os.path.join(PROJ, "results", "external_predictions.csv")]
found = next((c for c in cands if os.path.exists(c)), None)
if not found:
    print("[缺失] 两处都未找到 external_predictions.csv")
else:
    print("使用：", found)
    with io.open(found, encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        rows = list(r)
        print("行数：%d" % len(rows))
        print("列名：", r.fieldnames)
        if rows: print("首行：", rows[0])
        if "outcome" in (r.fieldnames or []):
            ev = sum(1 for x in rows if str(x["outcome"]).strip() in ("1","1.0","True"))
            print("事件数（outcome=1）：", ev, "  期望 88")

# ---------- 4. 仓库目录 ----------
line("4. 仓库目录检查")
print("REPO 存在：", os.path.isdir(REPO))
rep = os.path.join(REPO, "reproduce")
if not os.path.isdir(rep):
    os.makedirs(rep); print("已创建：", rep)
else:
    print("已存在：", rep)

line("自检结束。请把以上完整输出贴回。")