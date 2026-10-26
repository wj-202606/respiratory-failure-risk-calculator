# -*- coding: utf-8 -*-
import io, shutil
p = r"C:\Users\24339\Desktop\6.30呼吸衰竭\6.30呼衰\1.R数据预处理+变量筛选9.21\verify_frozen.py"
shutil.copy2(p, p + ".bak2_20261026")
s = io.open(p, encoding="utf-8").read()

old1 = '"intval_auc": (0.712, 0.0005)'
new1 = '"intval_auc": (0.7237, 0.0005)'
print("EXPECTED 命中:", s.count(old1)); s = s.replace(old1, new1)

old2 = '''    if "auc" in iv.columns:
        record("intval_auc", iv["auc"].mean(), "internal_validation_50imp.csv (mean)")'''
new2 = '''    if "auc" in iv.columns:
        record("intval_auc", iv["auc"].mean(), "internal_validation_50imp.csv (mean)")
    elif "Metric" in iv.columns and "Mean" in iv.columns:
        _m = iv.loc[iv["Metric"] == "AUC", "Mean"]
        record("intval_auc", float(_m.iloc[0]) if len(_m) else None,
               "internal_validation_50imp.csv (Metric=AUC, Mean)")'''
print("读取逻辑 命中:", s.count(old2)); s = s.replace(old2, new2)

io.open(p, "w", encoding="utf-8").write(s)
print("done")