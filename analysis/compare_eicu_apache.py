# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
import os

# 【请确认这里的路径是否正确】
PATH = r"D:\eicu数据库\EICU 2.0数据"

def delong_roc_test(y_true, y_pred1, y_pred2):
    n = len(y_true)
    pos = y_true == 1
    neg = y_true == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    
    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        T2 = np.empty(N, dtype=float)
        T2[J] = T
        return T2

    def fastDeLong(predictions_sorted_transposed, label_1_count):
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]
        
        tx = np.empty([k, m])
        ty = np.empty([k, n])
        tz = np.empty([k, m + n])
        for r in range(k):
            tx[r, :] = compute_midrank(positive_examples[r, :])
            ty[r, :] = compute_midrank(negative_examples[r, :])
            tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov

    preds = np.vstack([y_pred1, y_pred2])
    order = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    preds_sorted = preds[:, order]
    aucs, delongcov = fastDeLong(preds_sorted, label_1_count)
    
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, delongcov), l.T)).flatten()
    p_value = 2 * norm.sf(z)[0]
    return aucs[0], aucs[1], p_value

def main():
    print("1. 读取 eICU 预测结果...")
    be_df = pd.read_csv("results/eicu_predictions_be.csv")
    hco3_df = pd.read_csv("results/eicu_predictions_hco3.csv")
    
    print("2. 读取 APACHE 结果表...")
    apache_file = os.path.join(PATH, "apachePatientResult.csv.gz")
    if not os.path.exists(apache_file):
        print(f"找不到文件: {apache_file}")
        return
        
    apache = pd.read_csv(apache_file, 
                         usecols=["patientunitstayid", "apacheversion", "predictedhospitalmortality"])
    
    # 过滤出 IVa 版本，且预测死亡率非空
    apache = apache[(apache["apacheversion"] == "IVa") & (apache["predictedhospitalmortality"].notna())]
    apache = apache.drop_duplicates(subset=["patientunitstayid"], keep="first")
    apache["apache_pred_prob"] = apache["predictedhospitalmortality"] / 100.0
    
    for name, df in [("BE", be_df), ("HCO3", hco3_df)]:
        print(f"\n=== {name} 模型 vs APACHE IVa ===")
        merged = df.merge(apache[["patientunitstayid", "apache_pred_prob"]], 
                          on="patientunitstayid", how="inner")
        print(f"合并后可用样本量: n = {len(merged)} (原队列 n = {len(df)})")
        
        y = merged["outcome"].values
        p_model = merged["pred_prob"].values
        p_apache = merged["apache_pred_prob"].values
        
        auc_model = roc_auc_score(y, p_model)
        auc_apache = roc_auc_score(y, p_apache)
        auc1, auc2, p_val = delong_roc_test(y, p_model, p_apache)
        
        print(f"3变量模型 AUC: {auc_model:.3f}")
        print(f"APACHE IVa AUC: {auc_apache:.3f}")
        print(f"DeLong p-value: {p_val:.3e}")

if __name__ == "__main__":
    main()