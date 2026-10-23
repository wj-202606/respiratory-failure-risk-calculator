import pandas as pd
import numpy as np

# ======================
# 1. 开发队列（n = 2,028）
# ======================
train = pd.read_csv("results/traindata.csv")
val   = pd.read_csv("results/valdata.csv")
dev   = pd.concat([train, val], ignore_index=True)
print(f"Development cohort: n = {len(dev)}")

# ======================
# 2. 外部队列（n = 434）
# ======================
ext_ids = pd.read_csv("results/external_predictions.csv")["hadm_id"].values
print(f"External IDs (from external_predictions.csv): n = {len(ext_ids)}")
# ⚠️ mimic_wide_with_demo.csv 的 be_mmol_l 是真 BE（非 HCO3-）。
# 勿与 mimic_wide_hco3.csv 混淆（后者 be_mmol_l 存的是 HCO3-）。
mimic = pd.read_csv(r"D:\mimic_wide_with_demo.csv")
mimic = mimic[mimic["hadm_id"].isin(ext_ids)].copy()
print(f"External cohort after matching: n = {len(mimic)}")

# ======================
# 3. SMD 函数
# ======================
def smd_continuous(x1, x2):
    x1 = pd.to_numeric(x1, errors="coerce").dropna()
    x2 = pd.to_numeric(x2, errors="coerce").dropna()
    n1, n2 = len(x1), len(x2)
    m1, m2 = x1.mean(), x2.mean()
    s1, s2 = x1.std(ddof=1), x2.std(ddof=1)
    pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2))
    return (m1 - m2) / pooled if pooled > 0 else np.nan, m1, s1, m2, s2

def smd_binary(p1, p2):
    return (p1 - p2) / np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)

results = []

# ---- Age ----
smd_age, m1, s1, m2, s2 = smd_continuous(dev["age"], mimic["age"])
results.append(("Age, years, mean (SD)",
                f"{m1:.1f} ({s1:.1f})",
                f"{m2:.1f} ({s2:.1f})",
                f"{smd_age:.2f}"))

# ---- Male ----
# 开发队列 sex: 1 = male（假设），需确认
p_dev_male = (pd.to_numeric(dev["sex"], errors="coerce") == 1).mean()
p_ext_male = (pd.to_numeric(mimic["male"], errors="coerce") == 1).mean()
smd_male = smd_binary(p_dev_male, p_ext_male)
results.append(("Male, %",
                f"{p_dev_male*100:.1f}",
                f"{p_ext_male*100:.1f}",
                f"{smd_male:.2f}"))

# ---- ALB ----
smd_alb, m1, s1, m2, s2 = smd_continuous(dev["ALB"], mimic["alb_g_l"])
results.append(("ALB, g/L, mean (SD)",
                f"{m1:.2f} ({s1:.2f})",
                f"{m2:.2f} ({s2:.2f})",
                f"{smd_alb:.2f}"))

# ---- BUN ----
smd_bun, m1, s1, m2, s2 = smd_continuous(dev["BUN"], mimic["bun_mmol_l"])
results.append(("BUN, mmol/L, mean (SD)",
                f"{m1:.2f} ({s1:.2f})",
                f"{m2:.2f} ({s2:.2f})",
                f"{smd_bun:.2f}"))

# ---- BE ----
smd_be, m1, s1, m2, s2 = smd_continuous(dev["BE"], mimic["be_mmol_l"])
results.append(("BE, mmol/L, mean (SD)",
                f"{m1:.2f} ({s1:.2f})",
                f"{m2:.2f} ({s2:.2f})",
                f"{smd_be:.2f}"))

# ---- In-hospital mortality ----
p_dev_mort = pd.to_numeric(dev["Outcome"], errors="coerce").mean()
p_ext_mort = pd.to_numeric(pd.read_csv("results/external_predictions.csv")["outcome"], errors="coerce").mean()
smd_mort = smd_binary(p_dev_mort, p_ext_mort)
results.append(("In-hospital mortality, %",
                f"{p_dev_mort*100:.1f}",
                f"{p_ext_mort*100:.1f}",
                f"{smd_mort:.2f}"))

# ======================
# 4. 输出
# ======================
print("\n" + "="*90)
print(f"{'Variable':<32} {'Development':<22} {'External':<22} {'SMD':>6}")
print("="*90)
for var, dev_str, ext_str, smd_str in results:
    print(f"{var:<32} {dev_str:<22} {ext_str:<22} {smd_str:>6}")
print("="*90)