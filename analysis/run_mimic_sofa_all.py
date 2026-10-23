# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

# ================= 配置路径 =================
MIMIC_PATH = r"D:\mimic-iv-3.1"
RESULTS_DIR = "results"
OUT_SOFA = os.path.join(RESULTS_DIR, "mimic_sofa_24h.csv")

# ================= DeLong 检验函数 =================
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

# ================= 提取 SOFA =================
def extract_sofa():
    print("步骤 1: 读取 icustays 和 labevents...")
    icu = pd.read_csv(os.path.join(MIMIC_PATH, "icu", "icustays.csv.gz"), usecols=["hadm_id", "stay_id", "intime"])
    icu["intime"] = pd.to_datetime(icu["intime"])
    icu["outtime"] = icu["intime"] + pd.Timedelta(hours=24)
    
    labs = pd.read_csv(os.path.join(MIMIC_PATH, "hosp", "labevents.csv.gz"), 
                       usecols=["hadm_id", "itemid", "charttime", "valuenum"])
    labs = labs[labs["itemid"].isin([50821, 51265, 50885, 50912])]
    labs["charttime"] = pd.to_datetime(labs["charttime"])
    
    merged_labs = labs.merge(icu, on="hadm_id")
    merged_labs = merged_labs[(merged_labs["charttime"] >= merged_labs["intime"]) & 
                              (merged_labs["charttime"] <= merged_labs["outtime"])]
    
    sofa_scores = []
    
    resp = merged_labs[merged_labs["itemid"] == 50821].groupby("stay_id")["valuenum"].min().reset_index()
    resp["sofa_resp"] = pd.cut(resp["valuenum"], bins=[-np.inf, 100, 200, 300, 400, np.inf], labels=[4,3,2,1,0], right=False).astype(float)
    sofa_scores.append(resp[["stay_id", "sofa_resp"]])
    
    coag = merged_labs[merged_labs["itemid"] == 51265].groupby("stay_id")["valuenum"].min().reset_index()
    coag["sofa_coag"] = pd.cut(coag["valuenum"], bins=[-np.inf, 20, 50, 100, 150, np.inf], labels=[4,3,2,1,0], right=False).astype(float)
    sofa_scores.append(coag[["stay_id", "sofa_coag"]])
    
    liver = merged_labs[merged_labs["itemid"] == 50885].groupby("stay_id")["valuenum"].max().reset_index()
    liver["sofa_liver"] = pd.cut(liver["valuenum"], bins=[-np.inf, 1.2, 2.0, 6.0, 12.0, np.inf], labels=[0,1,2,3,4], right=False).astype(float)
    sofa_scores.append(liver[["stay_id", "sofa_liver"]])
    
    renal = merged_labs[merged_labs["itemid"] == 50912].groupby("stay_id")["valuenum"].max().reset_index()
    renal["sofa_renal"] = pd.cut(renal["valuenum"], bins=[-np.inf, 1.2, 2.0, 3.5, 5.0, np.inf], labels=[0,1,2,3,4], right=False).astype(float)
    sofa_scores.append(renal[["stay_id", "sofa_renal"]])

    print("步骤 2: 分块读取 chartevents 提取 MAP 和 GCS (不占内存)...")
    # 获取需要保留的 stay_id 列表，减少内存开销
    valid_stays = set(icu["stay_id"].values)
    
    # 存储提取的 chart 结果
    cardio_list = []
    cns_list = []
    
    chunksize = 2000000 # 每次读 200 万行
    iterator = pd.read_csv(os.path.join(MIMIC_PATH, "icu", "chartevents.csv.gz"), 
                           usecols=["stay_id", "charttime", "itemid", "valuenum"], 
                           chunksize=chunksize)
    
    for i, chunk in enumerate(iterator):
        # 过滤需要的 itemid
        chunk = chunk[chunk["itemid"].isin([220052, 225312, 223900])]
        if len(chunk) == 0:
            continue
        
        # 过滤出我们队列中的 stay_id
        chunk = chunk[chunk["stay_id"].isin(valid_stays)]
        if len(chunk) == 0:
            continue
            
        chunk["charttime"] = pd.to_datetime(chunk["charttime"])
        
        # 合并 intime 过滤 24h 内
        chunk_merged = chunk.merge(icu[["stay_id", "intime", "outtime"]], on="stay_id")
        chunk_merged = chunk_merged[(chunk_merged["charttime"] >= chunk_merged["intime"]) & 
                                    (chunk_merged["charttime"] <= chunk_merged["outtime"])]
        
        if len(chunk_merged) == 0:
            continue
            
        # 心血管 (MAP)
        cardio_part = chunk_merged[chunk_merged["itemid"].isin([220052, 225312])]
        if len(cardio_part) > 0:
            cardio_list.append(cardio_part[["stay_id", "valuenum"]])
            
        # 中枢 (GCS)
        cns_part = chunk_merged[chunk_merged["itemid"] == 223900]
        if len(cns_part) > 0:
            cns_list.append(cns_part[["stay_id", "valuenum"]])
            
        print(f"  - 已处理 {((i+1)*chunksize)/1000000:.0f} 万行...", flush=True)

    print("步骤 3: 汇总 SOFA...")
    
    # 处理心血管
    if cardio_list:
        cardio_all = pd.concat(cardio_list)
        cardio = cardio_all.groupby("stay_id")["valuenum"].min().reset_index()
        cardio["sofa_cardio"] = np.where(cardio["valuenum"] < 70, 1.0, 0.0)
        sofa_scores.append(cardio[["stay_id", "sofa_cardio"]])
    
    # 处理中枢
    if cns_list:
        cns_all = pd.concat(cns_list)
        cns = cns_all.groupby("stay_id")["valuenum"].min().reset_index()
        cns["sofa_cns"] = pd.cut(cns["valuenum"], bins=[-np.inf, 6, 10, 13, 15, np.inf], labels=[4,3,2,1,0], right=False).astype(float)
        sofa_scores.append(cns[["stay_id", "sofa_cns"]])

    # 合并所有评分
    final_sofa = icu[["hadm_id", "stay_id"]]
    for df in sofa_scores:
        final_sofa = final_sofa.merge(df, on="stay_id", how="left")
        
    final_sofa = final_sofa.fillna(0)
    final_sofa["sofa_max"] = final_sofa[["sofa_resp", "sofa_coag", "sofa_liver", "sofa_cardio", "sofa_cns", "sofa_renal"]].sum(axis=1)
    result = final_sofa.groupby("hadm_id")["sofa_max"].max().reset_index()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result.to_csv(OUT_SOFA, index=False)
    print(f"✅ SOFA 计算完成，保存至: {OUT_SOFA}")
    return result

# ================= 主程序 =================
def main():
    if not os.path.exists(OUT_SOFA):
        print("开始从本地 CSV 提取 SOFA...")
        extract_sofa()
    else:
        print("检测到已有 SOFA 文件，直接读取...")
        
    print("\n开始对比 3变量模型 vs SOFA...")
    sofa_df = pd.read_csv(OUT_SOFA)
    
    be_pred = pd.read_csv("results/external_predictions.csv")
    hco3_pred = pd.read_csv("results/external_predictions_hco3_v2.csv")
    
    for name, df in [("BE", be_pred), ("HCO3", hco3_pred)]:
        print(f"\n=== {name} 模型 vs 首日 SOFA ===")
        prob_col = "pred_prob" if "pred_prob" in df.columns else "p"
        
        merged = df.merge(sofa_df, on="hadm_id", how="inner")
        print(f"合并后可用样本量: n = {len(merged)} (原队列 n = {len(df)})")
        
        merged = merged.dropna(subset=["sofa_max", prob_col])
        
        y = merged["outcome"].values
        p_model = merged[prob_col].values
        p_sofa = merged["sofa_max"].values
        
        auc_model = roc_auc_score(y, p_model)
        auc_sofa = roc_auc_score(y, p_sofa)
        auc1, auc2, p_val = delong_roc_test(y, p_model, p_sofa)
        
        print(f"3变量模型 AUC: {auc_model:.3f}")
        print(f"SOFA 评分 AUC: {auc_sofa:.3f}")
        print(f"DeLong p-value: {p_val:.3e}")

if __name__ == "__main__":
    main()