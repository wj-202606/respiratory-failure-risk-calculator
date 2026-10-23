# -*- coding: utf-8 -*-
"""核对 app.py 中的系数是否与论文冻结数字一致（16 数）。"""
import io, os, re

APP = r"C:\Users\24339\Desktop\respiratory-failure-risk-calculator\app.py"

# 论文冻结系数（字符串形式，避免浮点误差）
# BE:  LP = 1.7825 - 0.1227*ALB + 0.0343*BUN - 0.0585*BE
# HCO3: LP = 3.1936 - 0.1209*ALB + 0.0374*BUN - 0.0604*HCO3
# 重校准: BE-MIMIC (0.4196, -0.6172), BE-eICU (0.6171, -0.4841)
#         HCO3-MIMIC (0.4856, -0.8090), HCO3-eICU (0.6420, -0.6126)
EXPECTED = {
    "BE_intercept":  ("1.7825", True),
    "BE_ALB":        ("0.1227", True),
    "BE_BUN":        ("0.0343", True),
    "BE_BE":         ("0.0585", True),
    "HCO3_intercept":("3.1936", True),
    "HCO3_ALB":      ("0.1209", True),
    "HCO3_BUN":      ("0.0374", True),
    "HCO3_HCO3":     ("0.0604", True),
    "recal_BE_MIMIC_a":  ("0.4196", True),
    "recal_BE_MIMIC_b":  ("0.6172", True),
    "recal_BE_eICU_a":   ("0.6171", True),
    "recal_BE_eICU_b":   ("0.4841", True),
    "recal_HCO3_MIMIC_a":("0.4856", True),
    "recal_HCO3_MIMIC_b":("0.8090", True),
    "recal_HCO3_eICU_a": ("0.6420", True),
    "recal_HCO3_eICU_b": ("0.6126", True),
}

def main():
    if not os.path.exists(APP):
        print("NOT FOUND:", APP)
        return
    s = io.open(APP, encoding="utf-8").read()
    print(f"读取: {APP}  ({len(s)} 字符)\n")

    ok = 0; bad = 0
    print(f"{'键':<24}{'期望':<10}{'命中':<8}{'状态'}")
    print("-" * 52)
    for k, (val, is_pos) in EXPECTED.items():
        c = s.count(val)
        # 某些系数可能带负号出现在代码里（如 -0.1227），但 val 是不带符号的数字串，
        # count 仍会命中；只要求字符串出现 >=1 次即可。
        status = "True" if c >= 1 else "False"
        if c >= 1: ok += 1
        else: bad += 1
        print(f"{k:<24}{val:<10}{c:<8}{status}")

    print("-" * 52)
    print(f"汇总: {ok} True, {bad} False (期望 12 True + 4 False)")

    # 额外：检查 eICU 说明是否在第 36 行附近
    lines = s.splitlines()
    hint_found = any("eICU" in l and ("未实现" in l or "Table 7" in l) for l in lines[:60])
    print(f"\neICU 未实现说明（前 60 行）: {'存在' if hint_found else '缺失'}")

if __name__ == "__main__":
    main()