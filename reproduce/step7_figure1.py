# -*- coding: utf-8 -*-
"""step7_figure1.py  流程图 n 核对（不做图，只核对每层数字）
"""
exp={"Development total (deaths)":"2,028 (226)","Training (deaths)":"1,420 (154)","Internal val (deaths)":"608 (72)",
     "MIMIC eligible":"2,456","MIMIC complete-case (deaths)":"434 (88)","MIMIC HCO3 complete (deaths)":"684 (109)",
     "eICU total ICU stays":"200,859","eICU inclusion met (deaths)":"5,099 (1,010)","eICU after range excl":"5,094",
     "eICU BE complete (deaths)":"1,505 (368)","eICU HCO3 complete (deaths)":"2,434 (531)"}
print("Figure 1 数字核对表（请与论文 Figure 1 legend 逐项比对）")
for k,v in exp.items(): print(f"  {k:<32} {v}")
print("\n注：eICU 5,099→5,094 为剔除 5 例实验室值超范围，死亡 1,010 不变。")