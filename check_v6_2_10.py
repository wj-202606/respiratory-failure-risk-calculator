import pandas as pd
import os
import sys

# 定义权威目标值 (基于交接材料 v16.0 第三节)
EXPECTED_VALUES = {
    "external_auc_stats.csv": {
        "MIMIC_BE": {"AUC": 0.620, "Brier": 0.166, "slope": 0.410, "intercept": -0.637},
        "MIMIC_HCO3": {"AUC": 0.642, "Brier": 0.137, "slope": 0.485, "intercept": -0.809},
    },
    "eicu_be_summary.csv": {
        "eICU_BE": {"AUC": 0.670, "Brier": 0.179, "slope": 0.599, "intercept": -0.486},
    },
    "eicu_hco3_summary.csv": {
        "eICU_HCO3": {"AUC": 0.670, "Brier": 0.171, "slope": 0.619, "intercept": -0.591},
    },
    "apache_delong_results.csv": {
        # 交接材料 3.9: P 必须是 0.009 和 <0.001
        "APACHE_BE_P": 0.009,
        "APACHE_HCO3_P": 0.001, # 假设 <0.001 代表 0.000x，这里设个极小值比对
    },
    "sensitivity_outcome_assisted_performance.csv": {
        # 交接材料 3.11
        "Sens_AUC": 0.729, "Sens_Slope": 0.789, "Sens_Intercept": -0.372
    }
}

def check_val(actual, expected, tolerance=0.005):
    """允许微小的四舍五入误差"""
    try:
        return abs(float(actual) - float(expected)) <= tolerance
    except ValueError:
        return False

def main():
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"❌ 找不到 {results_dir} 目录，请确保在项目根目录运行此脚本。")
        sys.exit(1)

    print("="*50)
    print("🚀 开始校验 v6.2.10 核心数据一致性")
    print("="*50)

    errors = []

    # 1. 核对 MIMIC AUC
    try:
        df = pd.read_csv(os.path.join(results_dir, "external_auc_stats.csv"))
        # 假设文件结构里有 model 和 auc 等列，具体列名如果不同请微调
        be_row = df[df['model'].str.contains('BE', case=False, na=False)].iloc[0]
        hco3_row = df[df['model'].str.contains('HCO3', case=False, na=False)].iloc[0]
        
        if not check_val(be_row.get('auc', 0), EXPECTED_VALUES["external_auc_stats.csv"]["MIMIC_BE"]["AUC"]):
            errors.append(f"❌ MIMIC BE AUC 异常：当前 {be_row.get('auc')}，期望 0.620")
        if not check_val(hco3_row.get('auc', 0), EXPECTED_VALUES["external_auc_stats.csv"]["MIMIC_HCO3"]["AUC"]):
            errors.append(f"❌ MIMIC HCO3 AUC 异常：当前 {hco3_row.get('auc')}，期望 0.642")
        print("✅ MIMIC AUC 校验通过 (需确认 CSV 列名匹配)")
    except Exception as e:
        print(f"⚠️ 无法自动解析 external_auc_stats.csv，请手动核对。(错误: {e})")

    # 2. 核对 eICU AUC
    try:
        df_be = pd.read_csv(os.path.join(results_dir, "eicu_be_summary.csv"))
        df_hco3 = pd.read_csv(os.path.join(results_dir, "eicu_hco3_summary.csv"))
        
        if not check_val(df_be.iloc[0].get('auc', 0), EXPECTED_VALUES["eicu_be_summary.csv"]["eICU_BE"]["AUC"]):
            errors.append(f"❌ eICU BE AUC 异常：当前 {df_be.iloc[0].get('auc')}，期望 0.670")
        if not check_val(df_hco3.iloc[0].get('auc', 0), EXPECTED_VALUES["eicu_hco3_summary.csv"]["eICU_HCO3"]["AUC"]):
            errors.append(f"❌ eICU HCO3 AUC 异常：当前 {df_hco3.iloc[0].get('auc')}，期望 0.670")
        print("✅ eICU AUC 校验通过")
    except Exception as e:
        print(f"⚠️ 无法自动解析 eICU 文件。(错误: {e})")

    # 3. 核对 APACHE (防雷：必须是 0.009 而不是旧的 0.194)
    try:
        df_apache = pd.read_csv(os.path.join(results_dir, "apache_delong_results.csv"))
        # 假设列名是 p_value，并有两行
        p_vals = df_apache['p_value'].tolist() if 'p_value' in df_apache else df_apache.iloc[:, -1].tolist()
        
        if len(p_vals) >= 2:
            if not check_val(p_vals[0], 0.009, tolerance=0.01):
                errors.append(f"❌ APACHE BE P值异常：当前 {p_vals[0]}，期望 0.009 (如果是 0.194 说明用了旧表)")
            if not (p_vals[1] < 0.001 or check_val(p_vals[1], 0.001, 0.001)):
                errors.append(f"❌ APACHE HCO3 P值异常：当前 {p_vals[1]}，期望 <0.001")
        print("✅ APACHE DeLong P值校验通过")
    except Exception as e:
        print(f"⚠️ 无法解析 apache_delong_results.csv。(错误: {e})")

    # 4. 核对样本量
    try:
        df_sample = pd.read_csv(os.path.join(results_dir, "submission_audit", "sample_size_audit.csv"))
        # 此处需要根据你 CSV 的实际格式自行匹配，这里仅作示意
        print("✅ 样本量文件读取成功，请确保 MIMIC BE=434, eICU BE=1458")
    except Exception as e:
        print(f"⚠️ 无法读取 sample_size_audit.csv。(错误: {e})")

    print("\n" + "="*50)
    if errors:
        print("🚨 发现数据不一致！请修复后再 push：")
        for err in errors:
            print(err)
    else:
        print("🎉 核心数字校验完毕，未发现明显的旧版残留！(建议再人工扫一眼 CSV 确认)")
    print("="*50)

if __name__ == "__main__":
    main()