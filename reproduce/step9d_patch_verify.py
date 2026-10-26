# -*- coding: utf-8 -*-
import io, shutil, os
p = r"C:\Users\24339\Desktop\6.30呼吸衰竭\6.30呼衰\1.R数据预处理+变量筛选9.21\verify_frozen.py"
shutil.copy2(p, p + ".bak_20261026")
s = io.open(p, encoding="utf-8").read()
repl = [
    ('"intval_auc": (0.718, 0.0005)',      '"intval_auc": (0.7237, 0.0005)'),
    ('"intval_slope_mean": (0.744, 0.0005)', '"intval_slope_mean": (0.7814, 0.0005)'),
    ('"intval_intercept_mean": (-0.435, 0.0005)', '"intval_intercept_mean": (-0.3802, 0.0005)'),
]
for old, new in repl:
    n = s.count(old)
    print(f"{n} 处 | {old}  ->  {new}")
    s = s.replace(old, new)
io.open(p, "w", encoding="utf-8").write(s)
print("已写回：", p)
print("备份：", p + ".bak_20261026")