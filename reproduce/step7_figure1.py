# -*- coding: utf-8 -*-
"""step7_figure1.py  流程图 n 核对（方案 B）"""
exp = {
 "Development total (deaths)": "2,028 (226)",
 "Training (deaths)": "1,420 (154)",
 "Internal val (deaths)": "608 (72)",
 "MIMIC eligible": "2,456",
 "MIMIC complete-case (deaths)": "434 (88)",
 "MIMIC HCO3 complete (deaths)": "684 (109)",
 "eICU total ICU stays": "200,859",
 "eICU inclusion met (deaths)": "4,696 (977)",
 "eICU after range excl (BE)": "4,693",
 "eICU after range excl (HCO3)": "4,691",
 "eICU BE complete (deaths)": "1,458 (363)",
 "eICU HCO3 complete (deaths)": "2,287 (521)"
}
print("Figure 1 数字核对表（请与论文 Figure 1 legend 逐项比对）")
for k,v in exp.items(): print(f"  {k:<32} {v}")
print("\n注：eICU 4,696 为 ICU 24h 存活 + 首次 ICU stay；经生理范围过滤后 BE 4,693 / HCO3 4,691。")