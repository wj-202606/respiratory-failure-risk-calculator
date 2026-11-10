# ============================================================
# R_analysis.R — development-cohort analysis pipeline (v5.4)
# Project: respiratory-failure-risk-calculator
# Repo:    https://github.com/wj-202606/respiratory-failure-risk-calculator
# Zenodo:  10.5281/zenodo.23300381 (concept)
# Inputs:  respiratory.csv; Outputs: results/, figures/, 论文图/, imputed_data/
#
# v5.4 更新（相对 v5.3）：
#   [1] 第 2 节读入后统一 HCO3 列名：HCO3_ -> HCO3
#       使 7b 节 HCO3_VARS、8b-1 系数文件、下游 Python 均使用 ASCII 列名 HCO3
#   [2] 第 5.6 节文案修正：predictorMatrix['Outcome', ] 行和提示改为"应 = 0"
#       与 v5.3 "Outcome 既不插补也不作预测因子"口径一致
#
# v5.3 更新（相对 v5.2）：
#   [1] 训练/验证插补统一采用不含 Outcome 的预测因子插补，避免验证结局泄漏
#   [2] 对关键结局、模型变量、数值型变量和系数文件增加严格断言
#   [3] 明确100次随机划分为固定预设模型的重复划分稳健性分析
#   [4] Bootstrap及单次插补图形输出明确标记为描述性/探索性
#
# v5.2 更新（相对 v5.1）：
#   [1] 第 6b 节新增：全开发队列（n = 2,028）基线，写 results/dev_full_baseline.csv
#       口径与 Table 1 "Development (n = 2,028)" 列同源（第 1 套插补 traindata+valdata）
#       供 Python make_eicu_supp_tables.py 的 DEV 字典运行时读取
#
# v5.1 更新（相对 v5.0）：
#   [1] 8b 节拆分为 8b-1（HCO3 三变量部署模型）与 8b-2（DD 四变量额外敏感性）
#   [2] 8b-1 写 external_validation_coef_hco3.csv / LP_formula_hco3.txt /
#       Rubin_Logistic_hco3.csv，供 Python 下游统一读取
#   [3] 8b-2 写 Rubin_Logistic_sens4.csv，明确标注不进部署链
#
# v5.0 更新（MICE 插补策略修正，对应审计 CRITICAL 1/2、HIGH 7、MODERATE 16）：
#   [1] 训练/验证集预测因子插补均不使用Outcome，Outcome本身不插补
#       （method['Outcome'] <- ""），避免验证集真实结局参与预测因子插补。
#   [2] 验证集插补：开发+验证拼接，ignore = 验证集，插补参数只在训练集估计
#       （冻结插补流程），不再在验证集内部独立估计插补参数。
#   [3] 唯一部署系数 = 50 套 Rubin 合并系数；内部验证也用该合并系数。
#   [4] 内部验证性能跨 50 套验证插补合并（均值 + SD + 分位数）。
#   [5] 100 次划分：FULL_PIPELINE_100SPLITS = TRUE，m = 50，每次重跑插补+合并。
#   [6] coef_first_vs_pooled_AUDITONLY.csv 保留为审计记录，不进报告链。
#
# v4.1 / v4.0 更新（保留）：见文件末尾历史注释。
# ============================================================

############################################################
######### 0. 安装 & 加载包 #########
############################################################
# install.packages(c("naniar","mice","tableone","ggplot2","lattice",
#                    "caret","autoReg","dplyr","Matrix","Boruta",
#                    "Hmisc","rms","glmnet","car","pROC","e1071","boot",
#                    "dcurves"))

library(naniar); library(mice); library(tableone); library(ggplot2)
library(lattice); library(caret); library(autoReg); library(dplyr)
library(Matrix); library(glmnet); library(Boruta); library(Hmisc)
library(rms); library(car); library(pROC); library(e1071); library(boot)

############################################################
######### 1. 设置工作目录 & 字体 #########
############################################################
if (nzchar(Sys.getenv("R_WD"))) setwd(Sys.getenv("R_WD"))

if (.Platform$OS.type == "windows")
  windowsFonts(Times = windowsFont("Times New Roman"))
par(family = "Times")

dir.create("results",      showWarnings = FALSE)
dir.create("figures",      showWarnings = FALSE)
dir.create("imputed_data", showWarnings = FALSE)
dir.create("论文图",        showWarnings = FALSE)

save_both <- function(plot_expr, filename_base, width = 8, height = 6) {
  pdf(file = paste0("figures/", filename_base, ".pdf"), width = width, height = height)
  eval(plot_expr)
  dev.off()
  png(file = paste0("figures/", filename_base, ".png"),
      width = width, height = height, units = "in", res = 300)
  eval(plot_expr)
  dev.off()
  cat("✅ 已保存 figures/", filename_base, ".pdf / .png\n", sep = "")
}

GLOBAL_SEED <- 123
set.seed(GLOBAL_SEED)
writeLines(paste0("Global seed: ", GLOBAL_SEED,
                  "\nOuter split seed: 123",
                  "\nImputation seed (train, v5.0): 123",
                  "\nImputation seed (frozen train+val, v5.0): 123",
                  "\nLASSO loop seed: 123 + imp_i",
                  "\n100-split seed: 1:100",
                  "\nBootstrap seed: 123"),
           "results/random_seeds.txt")

# v6.2：默认复用已完成的全流程100次划分，避免重跑并覆盖旧结果。
# 如确实需要重新运行100次全流程，将 RUN_100_SPLITS <- TRUE。
FULL_PIPELINE_100SPLITS <- TRUE
RUN_100_SPLITS <- TRUE
M_100SPLITS <- 50

############################################################
######### 2. 读数据 #########
############################################################
raw <- read.csv("respiratory.csv",
                fileEncoding = "GBK",
                na.strings = c("", "NA", "NULL", " ", "  ",
                               "不报告", "测不出", "未测", "未查", "缺失"),
                stringsAsFactors = FALSE)

# ★ v5.4：统一 HCO3 列名（respiratory.csv 原始列名为 HCO3_）
# 这样 7b 节的 HCO3_VARS、8b-1 系数文件、下游 Python 均使用 ASCII 列名 HCO3，
# 无需在各处做 HCO3_ / HCO3 的兼容重命名。
if ("HCO3_" %in% names(raw) && !("HCO3" %in% names(raw))) {
  names(raw)[names(raw) == "HCO3_"] <- "HCO3"
  cat("已将列名 HCO3_ 重命名为 HCO3\n")
}

cat("\n===== 原始数据维度 =====\n"); print(dim(raw))
cat("\n===== 原始列名 =====\n");     print(names(raw))

############################################################
######### 3. 数据预处理 #########
############################################################

cat_vars <- c("sex")
cat_vars <- cat_vars[cat_vars %in% names(raw)]
raw[cat_vars] <- lapply(raw[cat_vars], as.factor)
raw$Outcome <- suppressWarnings(as.integer(as.character(raw$Outcome)))

# 核心数据完整性检查：Outcome 必须为已知的二分类结局
if (!("Outcome" %in% names(raw))) stop("数据中缺少 Outcome 列。")
if (anyNA(raw$Outcome)) stop("Outcome 存在缺失；请在建模前按预设纳排标准处理，不能静默删除。")
if (!all(raw$Outcome %in% c(0L, 1L))) stop("Outcome 必须编码为 0/1。")
if (length(unique(raw$Outcome)) != 2L) stop("Outcome 必须同时包含 0 和 1。")
cat("\nOutcome 编码检查通过：\n")
print(table(raw$Outcome))

char_cols <- names(raw)[sapply(raw, is.character)]
cat("\n===== 读入后为 character 的列 =====\n"); print(char_cols)

for (v in char_cols) {
  x  <- trimws(raw[[v]])
  x[x == ""] <- NA
  xn <- suppressWarnings(as.numeric(x))
  if (sum(!is.na(xn)) >= 0.5 * length(xn)) {
    bad <- unique(x[is.na(xn) & !is.na(x)])
    if (length(bad) > 0)
      cat("[", v, "] 无法转数值的原值：", paste(head(bad, 20), collapse=", "), "\n")
    raw[[v]] <- xn
    cat("[", v, "] -> numeric，NA：", sum(is.na(xn)), "\n")
  }
}

char_to_num <- c("APTT","NEU","LYM","RDW","PLT","K","TCO2","BE","SaO2")
char_to_num <- char_to_num[char_to_num %in% names(raw)]
for (var in char_to_num) {
  raw[[var]] <- suppressWarnings(as.numeric(trimws(as.character(raw[[var]]))))
  cat(var, "-> NA：", sum(is.na(raw[[var]])), "\n")
}

stopifnot(is.numeric(raw$APTT))
cat("\nAPTT 类型：", class(raw$APTT), "\n")
print(summary(raw$APTT))

cat("\n===== 缺失情况 =====\n"); print(colSums(is.na(raw)))
str(raw)

p_miss <- raw %>%
  gg_miss_var(show_pct = TRUE) +
  theme_bw(base_size = 16, base_family = "Times")
ggsave("figures/missing_pattern.pdf", p_miss, width = 8, height = 8, dpi = 300)
ggsave("figures/missing_pattern.png", p_miss, width = 8, height = 8, dpi = 300)

missing_ratio_df <- data.frame(Variable = names(colMeans(is.na(raw))),
                               Missing_Ratio = colMeans(is.na(raw)))
write.csv(missing_ratio_df, "results/missing_ratio.csv",
          row.names = FALSE, quote = FALSE)

saveRDS(raw, "results/raw_processed.rds")

############################################################
######### 4. Little's MCAR 检验 #########
############################################################
cat("\n===== Little's MCAR 检验 =====\n")
mcar_input <- raw
mcar_input[] <- lapply(mcar_input, function(x)
  if (is.character(x)) as.factor(x) else x)
mcar_result <- mcar_test(mcar_input)
print(mcar_result)

if (mcar_result$p.value > 0.05) {
  cat("\nP > 0.05，不能拒绝 MCAR。\n")
} else {
  cat("\nP < 0.05，拒绝 MCAR；但多重插补在 MAR 假设下仍适用。\n")
}
write.csv(data.frame(statistic = mcar_result$statistic,
                     df        = mcar_result$df,
                     p.value   = mcar_result$p.value,
                     missing.patterns = mcar_result$missing.patterns),
          "results/Little_MCAR.csv", row.names = FALSE, quote = FALSE)

############################################################
######### 5. 7:3 划分 + v5.0 插补策略
############################################################
set.seed(123)
raw$Outcome <- as.integer(as.character(raw$Outcome))
inTrain <- createDataPartition(y = raw$Outcome, p = 0.7, list = FALSE)
raw_train <- raw[inTrain, ]
raw_val   <- raw[-inTrain, ]

cat("\n===== 外层划分 =====")
cat("train:", nrow(raw_train), " val:", nrow(raw_val), "\n")
cat("train events:", sum(raw_train$Outcome), " val events:", sum(raw_val$Outcome), "\n")

write.csv(data.frame(idx = inTrain), "results/train_indices.csv", row.names = FALSE)

# ============================================================
# 5.2 训练集插补（m=50）—— Outcome不插补且不作为插补预测因子
# ============================================================
cat("\n===== 5.2 训练集 MICE（Outcome不插补且不作为插补预测因子）=====\n")

if (any(is.na(raw_train$Outcome))) {
  cat("⚠️ 训练集 Outcome 有缺失，剔除", sum(is.na(raw_train$Outcome)), "行\n")
  raw_train <- raw_train[!is.na(raw_train$Outcome), ]
}
if (any(is.na(raw_val$Outcome))) {
  cat("⚠️ 验证集 Outcome 有缺失，剔除", sum(is.na(raw_val$Outcome)), "行\n")
  raw_val <- raw_val[!is.na(raw_val$Outcome), ]
}

set.seed(123)
pred_mat_train <- make.predictorMatrix(raw_train)
meth_train     <- make.method(raw_train)

meth_train["Outcome"] <- ""
# 重要：Outcome不插补，也不作为任何预测因子的插补预测变量。
# 这样训练阶段和验证阶段均不依赖验证集真实结局，符合预测时信息可得性。
pred_mat_train["Outcome", ] <- 0
pred_mat_train[, "Outcome"] <- 0
diag(pred_mat_train) <- 0

imp_train <- mice(raw_train, m = 50, seed = 123, print = FALSE,
                  method = meth_train,
                  predictorMatrix = pred_mat_train)

stopifnot(sum(imp_train$predictorMatrix["Outcome", ]) == 0)
stopifnot(sum(imp_train$predictorMatrix[, "Outcome"]) == 0)
stopifnot(imp_train$method["Outcome"] == "")

cat("✅ 训练集插补完成（m=50）\n")
cat("   predictorMatrix['Outcome', ] 行和 =",
    sum(imp_train$predictorMatrix["Outcome", ]), "（应 = 0）\n")
cat("   predictorMatrix[, 'Outcome'] 列和 =",
    sum(imp_train$predictorMatrix[, "Outcome"]), "（应 = 0）\n")
cat("   method['Outcome'] = '", imp_train$method["Outcome"], "'（应为空）\n", sep = "")

saveRDS(imp_train, "results/imp_train_m50_v5.rds")

# ============================================================
# 5.3 验证集插补（冻结流程）—— 开发+验证拼接，ignore = 验证集
# ============================================================
cat("\n===== 5.3 验证集 MICE（冻结开发集插补参数）=====\n")

all_data <- rbind(
  cbind(raw_train, .set = "train"),
  cbind(raw_val,   .set = "valid")
)
all_data$.set <- factor(all_data$.set, levels = c("train", "valid"))

set.seed(123)
pred_mat_all <- make.predictorMatrix(all_data)
meth_all     <- make.method(all_data)

meth_all["Outcome"] <- ""
meth_all[".set"]    <- ""

# Outcome既不被插补，也不参与任何预测因子的插补，防止验证集结局泄漏
pred_mat_all["Outcome", ] <- 0
pred_mat_all[, "Outcome"] <- 0
pred_mat_all[, ".set"] <- 0
pred_mat_all[".set", ] <- 0
diag(pred_mat_all) <- 0

imp_all <- mice(all_data, m = 50, seed = 123, print = FALSE,
                method = meth_all,
                predictorMatrix = pred_mat_all,
                ignore = (all_data$.set == "valid"))

imp_val <- filter(imp_all, .set == "valid")

cat("✅ 验证集插补完成（m=50，冻结开发集参数）\n")
cat("   参与参数估计的样本数（开发集）:",
    sum(all_data$.set == "train"), "\n")
cat("   被插补但不参与估计的样本数（验证集）:",
    sum(all_data$.set == "valid"), "\n")

saveRDS(imp_all, "results/imp_all_m50_v5.rds")
saveRDS(imp_val, "results/imp_val_m50_v5.rds")

# ============================================================
# 5.4 第 1 套仅用于描述性基线表
# ============================================================
df_train <- mice::complete(imp_all, action = 1) %>% filter(.set == "train") %>% select(-.set)
df_val   <- mice::complete(imp_all, action = 1) %>% filter(.set == "valid") %>% select(-.set)
traindata <- df_train
valdata   <- df_val

write.csv(traindata, "results/traindata_descriptive.csv", row.names = FALSE, quote = FALSE)
write.csv(valdata,   "results/valdata_descriptive.csv",   row.names = FALSE, quote = FALSE)

for (i in 1:50) {
  d <- mice::complete(imp_train, action = i)
  write.csv(d, sprintf("imputed_data/imputed_train_%02d.csv", i),
            row.names = FALSE)
}
cat("✅ 训练集 50 套插补数据已保存\n")

# ============================================================
# 5.6 核对
# ============================================================
cat("\n===== 5.6 验证集缺失与插补模型核对（v5.0）=====\n")

val_miss <- colMeans(is.na(raw_val[, c("ALB","BUN","BE","Outcome")]))
val_complete <- mean(complete.cases(raw_val[, c("ALB","BUN","BE")]))
val_n_missing <- sum(!complete.cases(raw_val[, c("ALB","BUN","BE")]))

cat("验证集 n =", nrow(raw_val), "\n")
cat("验证集缺失比例：\n"); print(round(val_miss, 4))
cat("三预测因子完整比例:", round(val_complete, 4), "\n")
cat("至少缺一个的人数:", val_n_missing, "\n")
cat("训练集 predictorMatrix Outcome 行和:", sum(imp_train$predictorMatrix["Outcome", ]), "（应 = 0）\n")
cat("训练集 predictorMatrix Outcome 列和:", sum(imp_train$predictorMatrix[, "Outcome"]), "（应 = 0）\n")
cat("训练集 method[Outcome]:", imp_train$method["Outcome"], "（应为空）\n")
cat("合并集 method[Outcome]:", imp_all$method["Outcome"], "（应为空）\n")
cat("合并集 ignore 中验证集行数:", sum(imp_all$ignore), "（应 = 验证集 n）\n")

imp_val_check <- data.frame(
  Item = c("val_n", "miss_ALB", "miss_BUN", "miss_BE", "miss_Outcome",
           "complete_case_prop", "n_missing_any",
           "train_predmat_outcome_row_sum", "train_predmat_outcome_col_sum",
           "train_method_Outcome", "all_method_Outcome", "all_ignore_n_valid"),
  Value = c(nrow(raw_val),
            round(val_miss["ALB"], 4), round(val_miss["BUN"], 4),
            round(val_miss["BE"], 4), round(val_miss["Outcome"], 4),
            round(val_complete, 4), val_n_missing,
            sum(imp_train$predictorMatrix["Outcome", ]),
            sum(imp_train$predictorMatrix[, "Outcome"]),
            imp_train$method["Outcome"],
            imp_all$method["Outcome"],
            sum(imp_all$ignore))
)
write.csv(imp_val_check, "results/imp_val_check_v5.csv",
          row.names = FALSE, quote = FALSE)
cat("✅ 已保存 results/imp_val_check_v5.csv\n")

############################################################
######### 6. 基线表（基于训练集第 1 套，仅描述性）
############################################################
cat("\n===== 6. 基线表（描述性，第 1 套插补）=====\n")
catVars  <- cat_vars
contVars <- setdiff(colnames(traindata), c("Outcome", catVars))
myVars   <- c(catVars, contVars)

noncenter <- c()
for (var in contVars) {
  if (is.numeric(traindata[[var]])) {
    sk <- skewness(traindata[[var]], na.rm = TRUE)
    if (abs(sk) > 1) noncenter <- c(noncenter, var)
  }
}
cat("\n非正态变量（|偏度|>1）：\n"); print(noncenter)

t1 <- CreateTableOne(vars = myVars, factorVars = catVars,
                     data = traindata, strata = "Outcome", addOverall = TRUE)
write.csv(print(t1, showAllLevels=TRUE, catDigits=2, contDigits=2, pDigits=3,
                nonnormal=noncenter, smd=TRUE, quote=FALSE, noSpaces=TRUE,
                printToggle=FALSE),
          "results/table1_train.csv", quote = FALSE)

t2 <- CreateTableOne(vars = myVars, factorVars = catVars,
                     data = valdata, strata = "Outcome", addOverall = TRUE)
write.csv(print(t2, showAllLevels=TRUE, catDigits=2, contDigits=2, pDigits=3,
                nonnormal=noncenter, smd=TRUE, quote=FALSE, noSpaces=TRUE,
                printToggle=FALSE),
          "results/table1_val.csv", quote = FALSE)

# ============================================================
# 6b. 全开发队列基线（供 Python DEV 字典运行时读取）
# ============================================================
# 口径：第 1 套插补的 traindata + valdata 合并，与 Table 1
#       "Development (n = 2,028)" 列同源。
# 输出：results/dev_full_baseline.csv
# 用途：make_eicu_supp_tables.py 的 load_dev_baseline() 读取，
#       用于 Table S7b 的 "Development (n = 2,028)" 列与 SMD。
cat("\n===== 6b. 全开发队列基线（n = 2,028）=====\n")

# 性别编码兜底：先打印 levels，便于核对
if ("sex" %in% names(traindata)) {
  cat("sex levels:", paste(levels(traindata$sex), collapse = " / "), "\n")
}

dev_full <- rbind(
  traindata %>% mutate(.set = "train"),
  valdata   %>% mutate(.set = "valid")
)

# 性别 male 判定（兼容 Male/male/M/1 等编码）
male_flag <- rep(NA_integer_, nrow(dev_full))
if ("sex" %in% names(dev_full)) {
  sex_chr <- tolower(trimws(as.character(dev_full$sex)))
  male_flag <- as.integer(sex_chr %in% c("male", "m", "1", "男"))
}

dev_summary <- data.frame(
  Variable = c("n", "age_mean", "age_sd", "male_pct",
               "alb_mean", "alb_sd", "bun_mean", "bun_sd",
               "be_mean", "be_sd", "hco3_mean", "hco3_sd", "mort_pct"),
  Value = c(
    nrow(dev_full),
    mean(dev_full$age, na.rm = TRUE),
    sd(dev_full$age,   na.rm = TRUE),
    100 * mean(male_flag, na.rm = TRUE),
    mean(dev_full$ALB, na.rm = TRUE),
    sd(dev_full$ALB,   na.rm = TRUE),
    mean(dev_full$BUN, na.rm = TRUE),
    sd(dev_full$BUN,   na.rm = TRUE),
    mean(dev_full$BE,  na.rm = TRUE),
    sd(dev_full$BE,    na.rm = TRUE),
    if ("HCO3" %in% names(dev_full)) mean(dev_full$HCO3, na.rm = TRUE) else NA_real_,
    if ("HCO3" %in% names(dev_full)) sd(dev_full$HCO3, na.rm = TRUE) else NA_real_,
    100 * mean(dev_full$Outcome == 1, na.rm = TRUE)
  )
)
write.csv(dev_summary, "results/dev_full_baseline.csv",
          row.names = FALSE, quote = FALSE)
print(dev_summary)

# 运行时断言：n 必须等于 traindata + valdata 行数
stopifnot(nrow(dev_full) == nrow(traindata) + nrow(valdata))
cat("✅ 已保存 results/dev_full_baseline.csv（n =", nrow(dev_full), "）\n")
cat("   male_pct =", round(100 * mean(male_flag, na.rm = TRUE), 2),
    "（应在 60–75 之间；若异常请检查 sex 编码）\n")

############################################################
######### 7. LASSO 变量筛选（在 imp_train 上）
############################################################
zscore_scale <- function(x) {
  if (sd(x, na.rm = TRUE) == 0) return(x)
  (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
}

lassodata <- traindata %>%
  select(-Outcome) %>%
  mutate(across(all_of(cat_vars), as.factor)) %>%
  mutate(across(where(is.numeric), zscore_scale)) %>%
  as.data.frame()

x <- data.matrix(lassodata)
y <- as.numeric(as.character(traindata$Outcome))

set.seed(123)
lasso    <- glmnet(x, y, family = "binomial", nlambda = 1000, alpha = 1)
lasso.cv <- cv.glmnet(x, y, alpha = 1, nfolds = 10, family = "binomial")

pdf("figures/LASSO_cv_curve.pdf", width=8, height=6)
plot(lasso.cv)
abline(v = log(lasso.cv$lambda.min),  col="black", lty=2)
abline(v = log(lasso.cv$lambda.1se), col="black", lty=2)
dev.off()
png("figures/LASSO_cv_curve.png", width=8, height=6, units="in", res=300)
plot(lasso.cv)
abline(v = log(lasso.cv$lambda.min),  col="black", lty=2)
abline(v = log(lasso.cv$lambda.1se), col="black", lty=2)
dev.off()

pdf("figures/LASSO_coef_path.pdf", width=8, height=6)
plot(lasso.cv$glmnet.fit, xvar = "lambda")
abline(v = log(lasso.cv$lambda.min),  col="black", lty=2)
abline(v = log(lasso.cv$lambda.1se), col="black", lty=2)
dev.off()
png("figures/LASSO_coef_path.png", width=8, height=6, units="in", res=300)
plot(lasso.cv$glmnet.fit, xvar = "lambda")
abline(v = log(lasso.cv$lambda.min),  col="black", lty=2)
abline(v = log(lasso.cv$lambda.1se), col="black", lty=2)
dev.off()

lasso_coefs_matrix <- as.matrix(coef(lasso.cv, s = "lambda.1se"))
write.csv(as.data.frame(lasso_coefs_matrix),
          "results/LASSO_coefs.csv", row.names = TRUE, quote = FALSE)

nonzero_vars <- rownames(lasso_coefs_matrix)[lasso_coefs_matrix[, 1] != 0]
nonzero_vars <- nonzero_vars[nonzero_vars != "(Intercept)"]
cat("\n===== LASSO 选出的变量（lambda.1se, 第 1 套）=====\n"); print(nonzero_vars)

coefs_min <- as.matrix(coef(lasso.cv, s = "lambda.min"))
vars_min  <- rownames(coefs_min)[coefs_min[, 1] != 0]
vars_min  <- vars_min[vars_min != "(Intercept)"]
cat("\n===== lambda.min 选出 =====\n"); print(vars_min)

lasso_vars_by_imp <- list()
for (imp_i in 1:50) {
  d_i <- mice::complete(imp_train, action = imp_i)
  d_i$Outcome <- as.numeric(as.character(d_i$Outcome))
  train_i <- d_i

  lassodata_i <- train_i %>%
    select(-Outcome) %>%
    mutate(across(all_of(cat_vars), as.factor)) %>%
    mutate(across(where(is.numeric), zscore_scale)) %>%
    as.data.frame()

  x_i <- data.matrix(lassodata_i)
  y_i <- as.numeric(as.character(train_i$Outcome))

  set.seed(123 + imp_i)
  lasso.cv_i <- cv.glmnet(x_i, y_i, alpha = 1, nfolds = 10, family = "binomial")
  coefs_i <- as.matrix(coef(lasso.cv_i, s = "lambda.1se"))
  vars_i <- rownames(coefs_i)[coefs_i[, 1] != 0]
  vars_i <- vars_i[vars_i != "(Intercept)"]
  lasso_vars_by_imp[[imp_i]] <- vars_i
}

all_vars <- unique(unlist(lasso_vars_by_imp))
freq_table <- data.frame(
  Variable = all_vars,
  Selected_Count = sapply(all_vars, function(v)
    sum(sapply(lasso_vars_by_imp, function(x) v %in% x)))
)
freq_table$Selected_Pct <- round(100 * freq_table$Selected_Count / 50, 1)
freq_table <- freq_table[order(-freq_table$Selected_Count), ]
print(freq_table)
write.csv(freq_table, "results/LASSO_vars_frequency.csv",
          row.names = FALSE, quote = FALSE)
cat("✅ 50 套 LASSO 稳定性分析完成\n")

############################################################
# 7b. 最终变量集
############################################################
MAIN_VARS <- c("ALB", "BUN", "BE")
HCO3_VARS <- c("ALB", "BUN", "HCO3")
DD_VARS   <- c("DD", "ALB", "BUN", "BE")

# 严格检查：禁止因列名错误或变量缺失而静默缩减模型
for (v in MAIN_VARS) {
  if (!v %in% names(traindata)) stop("BE主模型缺少变量：", v)
  if (!is.numeric(traindata[[v]])) stop("BE主模型变量必须为数值型：", v)
}
for (v in HCO3_VARS) {
  if (!v %in% names(traindata)) stop("HCO3模型缺少变量：", v)
  if (!is.numeric(traindata[[v]])) stop("HCO3模型变量必须为数值型：", v)
}
for (v in DD_VARS) {
  if (!v %in% names(traindata)) stop("DD扩展模型缺少变量：", v)
  if (!is.numeric(traindata[[v]])) stop("DD扩展模型变量必须为数值型：", v)
}

cat("\n===== 主模型变量（预设 3 变量）=====\n"); print(MAIN_VARS)
cat("\n===== HCO3 敏感性模型变量（3 变量）=====\n"); print(HCO3_VARS)
cat("\n===== DD 四变量扩展模型变量（额外敏感性）=====\n"); print(DD_VARS)

write.csv(data.frame(
  Model = c(rep("Main3_BE", length(MAIN_VARS)),
            rep("Sens3_HCO3", length(HCO3_VARS)),
            rep("Extra4_DD",  length(DD_VARS))),
  Variable = c(MAIN_VARS, HCO3_VARS, DD_VARS)
), "results/final_vars.csv", row.names = FALSE, quote = FALSE)

f_main <- as.formula(paste("Outcome ~", paste(MAIN_VARS, collapse = " + ")))
f_hco3 <- as.formula(paste("Outcome ~", paste(HCO3_VARS, collapse = " + ")))
f_dd   <- as.formula(paste("Outcome ~", paste(DD_VARS,   collapse = " + ")))

############################################################
######### 8. BE 主模型 Rubin 合并
############################################################
cat("\n===== 8. BE 主模型 Rubin 合并系数（唯一部署方程）=====\n")
cat("主模型公式：\n"); print(f_main)

fit_list <- with(imp_train, glm(
  as.formula(paste("Outcome ~", paste(MAIN_VARS, collapse = " + "))),
  family = binomial
))
pooled <- pool(fit_list, dfcom = nrow(traindata))
pooled_summary <- summary(pooled, conf.int = TRUE)
print(pooled_summary)
write.csv(pooled_summary, "results/Rubin_Logistic_main3.csv",
          row.names = FALSE, quote = FALSE)

format_p <- function(p) ifelse(p < 0.001, "<0.001", sprintf("%.3f", p))

pooled_or <- data.frame(
  Variable = pooled_summary$term,
  Beta     = round(pooled_summary$estimate, 4),
  SE       = round(pooled_summary$std.error, 4),
  OR       = round(exp(pooled_summary$estimate), 3),
  Lower_CI = round(exp(pooled_summary$`2.5 %`), 3),
  Upper_CI = round(exp(pooled_summary$`97.5 %`), 3),
  p_value  = format_p(pooled_summary$p.value)
)
print(pooled_or)
write.csv(pooled_or, "results/Rubin_OR_main3.csv",
          row.names = FALSE, quote = FALSE)

ext_coef <- data.frame(
  Term = pooled_summary$term,
  Beta = pooled_summary$estimate,
  SE   = pooled_summary$std.error
)
write.csv(ext_coef, "results/external_validation_coef.csv",
          row.names = FALSE, quote = FALSE)
cat("\n✅ BE 唯一部署系数已保存 results/external_validation_coef.csv\n")
print(ext_coef)

lp_formula <- paste0("LP = ",
                     round(pooled_summary$estimate[1], 4),
                     paste0(sprintf(" %+0.4f * %s",
                                    pooled_summary$estimate[-1],
                                    pooled_summary$term[-1]),
                            collapse = ""))
writeLines(lp_formula, "results/LP_formula_main3.txt")
cat("\n===== BE 部署 LP 公式 =====\n"); cat(lp_formula, "\n")

cat("\n===== FMI =====\n"); print(pooled$pooled)
write.csv(pooled$pooled, "results/FMI_main3.csv", row.names = FALSE, quote = FALSE)

Multi_log <- glm(f_main, data = traindata, family = binomial)
vif_values <- vif(Multi_log)
cat("\nVIF：\n"); print(round(vif_values, 3))
write.csv(data.frame(Variable = names(vif_values), VIF = round(vif_values, 3)),
          "results/VIF_main3.csv", row.names = FALSE, quote = FALSE)

############################################################
######### 8b-1. HCO3 三变量敏感性模型
############################################################
cat("\n===== 8b-1. HCO3 三变量敏感性模型 Rubin 合并 =====\n")
cat("HCO3 模型公式：\n"); print(f_hco3)

fit_list_hco3 <- with(imp_train, glm(
  as.formula(paste("Outcome ~", paste(HCO3_VARS, collapse = " + "))),
  family = binomial
))
pooled_hco3 <- pool(fit_list_hco3, dfcom = nrow(traindata))
pooled_hco3_summary <- summary(pooled_hco3, conf.int = TRUE)
print(pooled_hco3_summary)
write.csv(pooled_hco3_summary, "results/Rubin_Logistic_hco3.csv",
          row.names = FALSE, quote = FALSE)

pooled_or_hco3 <- data.frame(
  Variable = pooled_hco3_summary$term,
  Beta     = round(pooled_hco3_summary$estimate, 4),
  SE       = round(pooled_hco3_summary$std.error, 4),
  OR       = round(exp(pooled_hco3_summary$estimate), 3),
  Lower_CI = round(exp(pooled_hco3_summary$`2.5 %`), 3),
  Upper_CI = round(exp(pooled_hco3_summary$`97.5 %`), 3),
  p_value  = format_p(pooled_hco3_summary$p.value)
)
write.csv(pooled_or_hco3, "results/Rubin_OR_hco3.csv",
          row.names = FALSE, quote = FALSE)
print(pooled_or_hco3)

ext_coef_hco3 <- data.frame(
  Term = pooled_hco3_summary$term,
  Beta = pooled_hco3_summary$estimate,
  SE   = pooled_hco3_summary$std.error
)
write.csv(ext_coef_hco3, "results/external_validation_coef_hco3.csv",
          row.names = FALSE, quote = FALSE)
cat("\n✅ HCO3 部署系数已保存 results/external_validation_coef_hco3.csv\n")
print(ext_coef_hco3)

lp_formula_hco3 <- paste0("LP = ",
                          round(pooled_hco3_summary$estimate[1], 4),
                          paste0(sprintf(" %+0.4f * %s",
                                         pooled_hco3_summary$estimate[-1],
                                         pooled_hco3_summary$term[-1]),
                                 collapse = ""))
writeLines(lp_formula_hco3, "results/LP_formula_hco3.txt")
cat("\n===== HCO3 部署 LP 公式 =====\n"); cat(lp_formula_hco3, "\n")

cat("\n===== HCO3 FMI =====\n"); print(pooled_hco3$pooled)
write.csv(pooled_hco3$pooled, "results/FMI_hco3.csv",
          row.names = FALSE, quote = FALSE)

############################################################
######### 8b-2. DD 四变量扩展模型（额外敏感性，不进部署链）
############################################################
cat("\n===== 8b-2. DD 四变量扩展模型（额外敏感性）=====\n")
cat("⚠️ 此模型仅作额外敏感性分析，不进入部署链\n")
cat("DD 模型公式：\n"); print(f_dd)

fit_list_dd <- with(imp_train, glm(
  as.formula(paste("Outcome ~", paste(DD_VARS, collapse = " + "))),
  family = binomial
))
pooled_dd <- pool(fit_list_dd, dfcom = nrow(traindata))
pooled_dd_summary <- summary(pooled_dd, conf.int = TRUE)

write.csv(pooled_dd_summary, "results/Rubin_Logistic_sens4.csv",
          row.names = FALSE, quote = FALSE)

pooled_or_dd <- data.frame(
  Variable = pooled_dd_summary$term,
  Beta     = round(pooled_dd_summary$estimate, 4),
  SE       = round(pooled_dd_summary$std.error, 4),
  OR       = round(exp(pooled_dd_summary$estimate), 3),
  Lower_CI = round(exp(pooled_dd_summary$`2.5 %`), 3),
  Upper_CI = round(exp(pooled_dd_summary$`97.5 %`), 3),
  p_value  = format_p(pooled_dd_summary$p.value)
)
write.csv(pooled_or_dd, "results/Rubin_OR_sens4.csv",
          row.names = FALSE, quote = FALSE)
print(pooled_or_dd)

cat("\n✅ DD 四变量扩展模型完成（额外敏感性，不进部署链）\n")

############################################################
######### 9. 稳健性分析（在 imp_train 上）
############################################################

coef_list <- lapply(1:50, function(i) {
  d <- mice::complete(imp_train, action = i)
  coef(glm(f_main, data = d, family = binomial))
})
coef_df <- do.call(rbind, coef_list)
coef_summary_table <- data.frame(
  Variable = colnames(coef_df),
  Mean   = round(colMeans(coef_df), 4),
  SD     = round(apply(coef_df, 2, sd), 4),
  Median = round(apply(coef_df, 2, median), 4),
  Q25    = round(apply(coef_df, 2, quantile, 0.25), 4),
  Q75    = round(apply(coef_df, 2, quantile, 0.75), 4),
  Min    = round(apply(coef_df, 2, min), 4),
  Max    = round(apply(coef_df, 2, max), 4)
)
print(coef_summary_table)
write.csv(coef_summary_table, "results/coef_50imp.csv",
          row.names = FALSE, quote = FALSE)

boot_coef <- function(data, indices) {
  d <- data[indices, ]
  coef(glm(f_main, data = d, family = binomial))
}
set.seed(123)
# 描述性Bootstrap：基于第1套完成数据集，不传播多重插补不确定性。
# 不将其解释为Rubin合并模型的正式置信区间。
boot_result <- boot(traindata, boot_coef, R = 1000)

boot_ci <- data.frame()
for (i in 1:length(boot_result$t0)) {
  ci <- boot.ci(boot_result, type = "perc", index = i)
  boot_ci <- rbind(boot_ci, data.frame(
    Variable     = names(boot_result$t0)[i],
    Coefficient  = round(boot_result$t0[i], 4),
    Bootstrap_SE = round(sd(boot_result$t[[i]]), 4),
    Lower_CI     = round(ci$percent[4], 4),
    Upper_CI     = round(ci$percent[5], 4)
  ))
}
print(boot_ci)
write.csv(boot_ci, "results/bootstrap_ci_single_imputation_exploratory.csv",
          row.names = FALSE, quote = FALSE)

for (target_var in MAIN_VARS) {
  idx <- which(names(boot_result$t0) == target_var)
  if (length(idx) == 0) next

  boot_vals <- boot_result$t[, idx]
  ci_t <- boot.ci(boot_result, type = "perc", index = idx)

  draw_boot_hist <- function() {
    par(mar = c(5, 5, 4, 2) + 0.1)
    hist(boot_vals, breaks = 40,
         main = paste("Bootstrap distribution of", target_var),
         xlab = paste(target_var, "coefficient"),
         col = "lightblue", border = "white",
         cex.main = 1.2, cex.lab = 1.1)
    abline(v = 0,                    col = "red",       lwd = 2, lty = 2)
    abline(v = boot_result$t0[idx],  col = "blue",      lwd = 2)
    abline(v = ci_t$percent[4],      col = "darkgreen", lwd = 2, lty = 3)
    abline(v = ci_t$percent[5],      col = "darkgreen", lwd = 2, lty = 3)
    legend("topright",
           legend = c("0",
                      paste0("Mean = ", round(boot_result$t0[idx], 3)),
                      paste0("95% CI [", round(ci_t$percent[4], 3),
                             ", ", round(ci_t$percent[5], 3), "]")),
           col = c("red", "blue", "darkgreen"),
           lwd = 2, lty = c(2, 1, 3),
           bty = "n", cex = 0.9)
  }

  pdf(paste0("figures/Bootstrap_", target_var, "_dist.pdf"), width=8, height=6)
  draw_boot_hist(); dev.off()

  png(paste0("figures/Bootstrap_", target_var, "_dist.png"),
      width=8, height=6, units="in", res=300)
  draw_boot_hist(); dev.off()
}
cat("✅ Bootstrap 分布图已保存\n")

############################################################
######### 10. 模型评估：训练集 + 内部验证（统一系数）
############################################################

coef_deploy <- pooled_summary$estimate
names(coef_deploy) <- pooled_summary$term
names(coef_deploy)[names(coef_deploy) == "(Intercept)"] <- "Intercept"

predict_lp <- function(dat, coef_vec, vars = MAIN_VARS) {
  lp <- coef_vec["Intercept"]
  for (v in vars) lp <- lp + coef_vec[v] * dat[[v]]
  as.numeric(lp)
}

cat("\n===== 10.2 训练集性能（合并系数，50 套插补）=====\n")
train_auc_list   <- c()
train_brier_list <- c()

for (i in 1:50) {
  d_i <- mice::complete(imp_train, action = i)
  y_i <- as.numeric(as.character(d_i$Outcome))
  p_i <- plogis(predict_lp(d_i, coef_deploy))
  train_auc_list   <- c(train_auc_list,   as.numeric(auc(y_i, p_i, direction = "<")))
  train_brier_list <- c(train_brier_list, mean((p_i - y_i)^2))
}
train_auc   <- mean(train_auc_list)
train_brier <- mean(train_brier_list)
cat(sprintf("Train AUC   (mean ± SD): %.4f ± %.4f\n", mean(train_auc_list),   sd(train_auc_list)))
cat(sprintf("Train Brier (mean ± SD): %.4f ± %.4f\n", mean(train_brier_list), sd(train_brier_list)))

cat("\n===== 10.3 内部验证（合并系数 × 50 套验证插补）=====\n")
auc_list    <- c()
brier_list  <- c()
slope_list  <- c()
int_list    <- c()

for (i in 1:50) {
  val_i <- mice::complete(imp_val, action = i)
  val_i$Outcome <- as.numeric(as.character(val_i$Outcome))

  lp_i_val <- predict_lp(val_i, coef_deploy)
  pred_i   <- plogis(lp_i_val)
  y_i      <- val_i$Outcome

  auc_list <- c(auc_list, as.numeric(auc(y_i, pred_i, direction = "<")))
  brier_list <- c(brier_list, mean((pred_i - y_i)^2))

  eps <- 1e-6
  lp_logit <- log(pmax(pmin(pred_i, 1-eps), eps) /
                  (1 - pmax(pmin(pred_i, 1-eps), eps)))
  cal_i <- glm(y_i ~ lp_logit, family = binomial)
  slope_list <- c(slope_list, coef(cal_i)[2])
  int_list   <- c(int_list,   coef(cal_i)[1])
}

cat(sprintf("Val AUC      (mean ± SD): %.4f ± %.4f\n", mean(auc_list),   sd(auc_list)))
cat(sprintf("Val Brier    (mean ± SD): %.4f ± %.4f\n", mean(brier_list), sd(brier_list)))
cat(sprintf("Val Cal slope(mean ± SD): %.4f ± %.4f\n", mean(slope_list), sd(slope_list)))
cat(sprintf("Val Cal int  (mean ± SD): %.4f ± %.4f\n", mean(int_list),   sd(int_list)))

val_summary <- data.frame(
  Metric = c("AUC", "Brier", "Cal_slope", "Cal_intercept"),
  Mean   = c(mean(auc_list), mean(brier_list), mean(slope_list), mean(int_list)),
  SD     = c(sd(auc_list),   sd(brier_list),   sd(slope_list),   sd(int_list)),
  Q25    = c(quantile(auc_list, .25), quantile(brier_list, .25),
             quantile(slope_list, .25), quantile(int_list, .25)),
  Q75    = c(quantile(auc_list, .75), quantile(brier_list, .75),
             quantile(slope_list, .75), quantile(int_list, .75))
)
write.csv(val_summary, "results/internal_validation_50imp_v5.csv",
          row.names = FALSE, quote = FALSE)

val_pred <- plogis(predict_lp(valdata, coef_deploy))
valdata$Outcome_num <- as.numeric(as.character(valdata$Outcome))
val_roc  <- roc(valdata$Outcome_num, val_pred)
val_auc  <- as.numeric(auc(val_roc))
val_brier <- mean((val_pred - valdata$Outcome_num)^2)
cat("\n===== 第 1 套描述性（仅绘图）=====\n")
cat("AUC =", round(val_auc, 4), " Brier =", round(val_brier, 4), "\n")

train_pred <- plogis(predict_lp(traindata, coef_deploy))
y_train <- as.numeric(as.character(traindata$Outcome))
train_roc <- roc(y_train, train_pred)
train_auc_single <- as.numeric(auc(train_roc))
train_brier_single <- mean((train_pred - y_train)^2)

brier_train_naive <- train_brier_single
brier_val_naive   <- val_brier
mae_train         <- mean(abs(train_pred - y_train))
mae_val           <- mean(abs(val_pred   - valdata$Outcome_num))

write.csv(data.frame(
  Metric = c("Train_AUC_single", "Train_Brier_single",
             "Val_AUC_single",   "Val_Brier_single",
             "MAE_train",        "MAE_val"),
  Value  = c(train_auc_single, train_brier_single,
             val_auc, val_brier,
             mae_train, mae_val)
), "results/brier_mae_check_v5.csv", row.names = FALSE, quote = FALSE)

cat("\n===== 10.6 审计记录：第 1 套 glm vs 合并系数 =====\n")

fit_first <- glm(f_main, data = traindata, family = binomial)
coef_first <- coef(fit_first)
coef_pooled <- pooled_summary$estimate
names(coef_pooled) <- pooled_summary$term

all_terms <- union(names(coef_first), names(coef_pooled))
stopifnot(all(all_terms %in% names(coef_first)))
stopifnot(all(all_terms %in% names(coef_pooled)))

cmp <- data.frame(
  Term   = all_terms,
  First  = round(coef_first[all_terms], 4),
  Pooled = round(coef_pooled[all_terms], 4),
  Diff   = round(coef_first[all_terms] - coef_pooled[all_terms], 4),
  row.names = NULL
)
print(cmp)
write.csv(cmp, "results/coef_first_vs_pooled_AUDITONLY.csv", row.names = FALSE, quote = FALSE)

cat("\n最大绝对差:", round(max(abs(cmp$Diff), na.rm = TRUE), 4), "\n")
cat("⚠️ 此文件仅供审计核对，不再进入报告链。\n")

############################################################
######### 11. 100 次随机划分 AUC（v6.2：默认复用已有全流程结果）
############################################################
cat("\n===== 11. 100 次随机划分 AUC =====\n")
cat("RUN_100_SPLITS =", RUN_100_SPLITS, "\n")
cat("FULL_PIPELINE_100SPLITS =", FULL_PIPELINE_100SPLITS, "\n")
cat("m (per split) =", M_100SPLITS, "\n")

auc_file <- "results/auc_100splits_v5.csv"
auc_results <- numeric(0)
direction_audit <- data.frame(
  split = integer(), imp = integer(),
  auc_fixed = numeric(), auc_auto = numeric(),
  flipped = logical()
)

if (RUN_100_SPLITS) {
  if (!FULL_PIPELINE_100SPLITS) {
    stop("v6.2正式分析不允许用单插补降级版覆盖全流程100次结果。请将FULL_PIPELINE_100SPLITS设为TRUE。")
  }
  cat("运行完整100次随机划分：每次重新进行m=", M_100SPLITS, "套插补和Rubin合并。\n", sep="")
  for (i in 1:100) {
    set.seed(i)
    idx_i <- createDataPartition(y = raw$Outcome, p = 0.7, list = FALSE)
    tr_i <- raw[idx_i, ]; va_i <- raw[-idx_i, ]

    pred_i <- make.predictorMatrix(tr_i)
    meth_i <- make.method(tr_i)
    meth_i["Outcome"] <- ""
    pred_i["Outcome", ] <- 0
    pred_i[, "Outcome"] <- 0
    diag(pred_i) <- 0

    set.seed(1000 + i)
    imp_i <- mice(tr_i, m = M_100SPLITS, seed = 1000 + i, print = FALSE,
                  method = meth_i, predictorMatrix = pred_i)
    fit_list_i <- lapply(1:M_100SPLITS, function(k) {
      d_k <- mice::complete(imp_i, action = k)
      glm(f_main, data = d_k, family = binomial)
    })
    pooled_i <- mice::pool(mice::as.mira(fit_list_i))
    coef_i <- summary(pooled_i)$estimate
    names(coef_i) <- summary(pooled_i)$term
    names(coef_i)[names(coef_i) == "(Intercept)"] <- "Intercept"

    all_i <- rbind(cbind(tr_i, .set = "train"), cbind(va_i, .set = "valid"))
    all_i$.set <- factor(all_i$.set, levels = c("train", "valid"))
    pred_all_i <- make.predictorMatrix(all_i)
    meth_all_i <- make.method(all_i)
    meth_all_i["Outcome"] <- ""
    meth_all_i[".set"] <- ""
    pred_all_i["Outcome", ] <- 0
    pred_all_i[, "Outcome"] <- 0
    pred_all_i[, ".set"] <- 0
    pred_all_i[".set", ] <- 0
    diag(pred_all_i) <- 0
    imp_all_i <- mice(all_i, m = M_100SPLITS, seed = 2000 + i, print = FALSE,
                      method = meth_all_i, predictorMatrix = pred_all_i,
                      ignore = (all_i$.set == "valid"))
    va_imp_i <- filter(imp_all_i, .set == "valid")

    auc_m <- numeric(0)
    for (k in 1:M_100SPLITS) {
      va_k <- mice::complete(va_imp_i, action = k)
      y_k <- as.numeric(as.character(va_k$Outcome))
      p_k <- plogis(predict_lp(va_k, coef_i))
      auc_fixed_val <- as.numeric(auc(y_k, p_k, direction = "<"))
      auc_auto_val  <- as.numeric(auc(y_k, p_k, direction = "auto"))
      auc_m <- c(auc_m, auc_fixed_val)
      direction_audit <- rbind(direction_audit, data.frame(
        split = i, imp = k,
        auc_fixed = auc_fixed_val,
        auc_auto  = auc_auto_val,
        flipped   = abs(auc_fixed_val - auc_auto_val) > 1e-9
      ))
    }
    auc_results <- c(auc_results, mean(auc_m))

    if (i %% 5 == 0) {
      cat("  split", i, "done, AUC =", round(mean(auc_m), 4), "\n")

      # checkpoint: 保存已完成的部分结果（不覆盖最终文件）
      tryCatch({
        write.csv(
          data.frame(
            Run = seq_along(auc_results),
            AUC = auc_results
          ),
          "results/auc_100splits_checkpoint.csv",
          row.names = FALSE
        )
        write.csv(
          direction_audit,
          "results/auc_100splits_direction_checkpoint.csv",
          row.names = FALSE
        )
        cat("    [checkpoint] saved", i, "/100\n")
      }, error = function(e) {
        cat("    [checkpoint] FAILED:", conditionMessage(e), "\n")
      })
    }
  }
  write.csv(direction_audit,
            "results/auc_100splits_direction_audit.csv",
            row.names = FALSE)
  cat("  direction audit saved:", nrow(direction_audit), "rows, flipped =",
      sum(direction_audit$flipped), "\n")
  if (length(auc_results) != 100 || any(!is.finite(auc_results))) {
    stop("100次全流程重复划分未完整完成，拒绝写出结果。")
  }
  write.csv(data.frame(Run = seq_along(auc_results), AUC = round(auc_results, 4),
                       Mode = "full_pipeline"), auc_file,
            row.names = FALSE, quote = FALSE)
  writeLines(c(
    "Mode: repeated_split_fixed_prespecified_model",
    "Imputation repeated within each split: YES",
    paste0("m per split: ", M_100SPLITS),
    paste0("Median AUC: ", round(median(auc_results), 4)),
    paste0("IQR: ", round(quantile(auc_results, .25), 4), " - ", round(quantile(auc_results, .75), 4)),
    paste0("SD: ", round(sd(auc_results), 4)),
    "",
    "The imputation and estimation pipeline for the prespecified fixed three-variable model was repeated within each random split; predictor selection was not repeated."
  ), "results/auc_100splits_legend_v5.txt")
} else {
  if (!file.exists(auc_file)) {
    stop("RUN_100_SPLITS=FALSE，但未找到既有 results/auc_100splits_v5.csv。请先恢复此前完成的全流程100次结果，或将RUN_100_SPLITS设为TRUE。")
  }
  old_auc <- read.csv(auc_file, stringsAsFactors = FALSE)
  if (!all(c("AUC", "Mode") %in% names(old_auc)) || nrow(old_auc) != 100 ||
      any(!is.finite(old_auc$AUC)) || any(old_auc$Mode != "full_pipeline")) {
    stop("既有 auc_100splits_v5.csv 不符合完整100次全流程结果契约；为防止误用，停止运行且不覆盖该文件。")
  }
  auc_results <- as.numeric(old_auc$AUC)
  cat("复用既有完整全流程100次结果；本次不重新计算、不覆盖 auc_100splits_v5.csv。\n")
}

med_auc <- median(auc_results)
q25_auc <- as.numeric(quantile(auc_results, .25))
q75_auc <- as.numeric(quantile(auc_results, .75))
cat("中位数 =", round(med_auc, 4), " IQR =", round(q25_auc, 4), "~",
    round(q75_auc, 4), " SD =", round(sd(auc_results), 4), "\n")

save_auc_chest <- function(path_pdf) {
  cairo_pdf(path_pdf, width = 7, height = 5, family = "Arial")
  par(mar = c(4.5, 4.5, 1.5, 1.0) + 0.1, family = "Arial",
      cex.lab = 1.1, cex.axis = 0.95)
  hist(auc_results, breaks = 20,
       xlab = "AUC", ylab = "Frequency",
       col = "#9ECAE1", border = "white", main = "")
  abline(v = med_auc, col = "black", lwd = 1.8, lty = 2)
  legend("topright",
         legend = sprintf("Median = %.3f (IQR %.3f-%.3f)",
                          med_auc, q25_auc, q75_auc),
         col = "black", lty = 2, lwd = 1.8, bty = "n", cex = 0.9)
  dev.off()
}
tryCatch({
  save_auc_chest("论文图/FigureS5_AUC_distribution.pdf")
  cat("✅ 已保存 论文图/FigureS5_AUC_distribution.pdf\n")
}, error = function(e) {
  cat("⚠️ cairo_pdf 不可用，改用 pdf()\n")
  pdf("论文图/FigureS5_AUC_distribution.pdf", width = 7, height = 5)
  hist(auc_results, breaks = 20, xlab = "AUC", ylab = "Frequency",
       col = "#9ECAE1", border = "white", main = "")
  abline(v = med_auc, col = "black", lwd = 1.8, lty = 2)
  legend("topright",
         legend = sprintf("Median = %.3f (IQR %.3f-%.3f)", med_auc, q25_auc, q75_auc),
         col = "black", lty = 2, lwd = 1.8, bty = "n", cex = 0.9)
  dev.off()
})

save_both(quote({
  hist(auc_results, breaks=20,
       main = "AUC across 100 random splits",
       xlab = "AUC", col="lightblue", border="white")
  abline(v = median(auc_results), col="red", lwd=2, lty=2)
  legend("topright", legend = paste0("Median = ", round(median(auc_results),3)),
         col="red", lty=2, lwd=2, bty="n")
}), "AUC_distribution", width=8, height=6)

############################################################
######### 12. ROC + 校准曲线（第 1 套绘图）
############################################################
save_both(quote({
  plot(train_roc, col="blue", main="ROC Curves")
  plot(val_roc, col="red", add=TRUE)
  legend("bottomright",
         legend = c(paste0("Train AUC = ", round(train_auc_single, 3)),
                    paste0("Valid AUC = ", round(val_auc, 3))),
         col = c("blue","red"), lwd = 2)
}), "ROC_curves", width=8, height=8)

saveRDS(list(train_roc = train_roc, val_roc = val_roc,
             train_auc = train_auc_single, val_auc = val_auc,
             coef_deploy = coef_deploy),
        "results/roc_objects_v5.rds")

set.seed(123)
fit_lrm_desc <- lrm(f_main, data = traindata, x = TRUE, y = TRUE)
cal_train <- calibrate(fit_lrm_desc, method = "boot", B = 1000)

save_both(quote({
  plot(cal_train, main = "Calibration Curve - Training (Bootstrap, B = 1000)",
       xlab = "Predicted Probability",
       ylab = "Observed Probability")
  abline(0, 1, col = "gray", lty = 2)
}), "calibration_curve_train", width=8, height=8)

save_both(quote({
  cal_df <- data.frame(p = val_pred, y = valdata$Outcome_num)
  cal_loess <- loess(y ~ p, data = cal_df, span = 0.75)
  grid_p <- seq(min(cal_df$p), max(cal_df$p), length.out = 200)
  pred_obj <- predict(cal_loess, newdata = data.frame(p = grid_p), se = TRUE)
  pred_fit <- pred_obj$fit
  pred_se  <- pred_obj$se.fit

  plot(cal_df$p, cal_df$y,
       xlab = "Predicted Probability", ylab = "Observed Probability",
       main = "Calibration - Validation Set",
       col = rgb(0, 0, 1, 0.25), pch = 16, cex = 0.6,
       ylim = c(0, 1), xlim = c(0, 1))
  polygon(c(grid_p, rev(grid_p)),
          c(pred_fit - 1.96 * pred_se, rev(pred_fit + 1.96 * pred_se)),
          col = rgb(1, 0, 0, 0.15), border = NA)
  lines(grid_p, pred_fit, col = "red", lwd = 2)
  abline(0, 1, col = "gray", lty = 2)
  legend("bottomright",
         legend = c("Loess smooth", "95% CI", "Ideal"),
         col = c("red", rgb(1, 0, 0, 0.3), "gray"),
         lty = c(1, NA, 2), lwd = c(2, NA, 2),
         fill = c(NA, rgb(1, 0, 0, 0.3), NA), border = NA, bty = "n")
}), "calibration_curve_val", width=8, height=8)

############################################################
######### 13. 汇总输出（v5.4）
############################################################
summary_out <- data.frame(
  Metric = c("Train_AUC_50imp_mean", "Train_AUC_50imp_sd",
             "Train_Brier_50imp_mean", "Train_Brier_50imp_sd",
             "Val_AUC_50imp_mean", "Val_AUC_50imp_sd",
             "Val_AUC_50imp_q25",  "Val_AUC_50imp_q75",
             "Val_Brier_50imp_mean", "Val_Brier_50imp_sd",
             "Val_slope_50imp_mean", "Val_slope_50imp_sd",
             "Val_intercept_50imp_mean", "Val_intercept_50imp_sd",
             "Val_AUC_single_desc", "Val_Brier_single_desc",
             "AUC_100split_median", "AUC_100split_IQR_lo",
             "AUC_100split_IQR_hi", "AUC_100split_mode",
             "dev_full_n"),
  Value = c(mean(train_auc_list), sd(train_auc_list),
            mean(train_brier_list), sd(train_brier_list),
            mean(auc_list), sd(auc_list),
            quantile(auc_list, .25), quantile(auc_list, .75),
            mean(brier_list), sd(brier_list),
            mean(slope_list), sd(slope_list),
            mean(int_list), sd(int_list),
            val_auc, val_brier,
            med_auc, q25_auc, q75_auc,
            ifelse(FULL_PIPELINE_100SPLITS, 1, 0),
            nrow(dev_full))
)
write.csv(summary_out, "results/summary_for_manuscript_v5.csv",
          row.names = FALSE, quote = FALSE)
print(summary_out)

check_summary <- data.frame(
  Item = c("val_n", "val_n_missing_any",
           "train_predmat_outcome_row_sum", "train_predmat_outcome_col_sum",
           "train_method_Outcome", "all_ignore_n_valid",
           "coef_max_abs_diff_first_vs_pooled",
           "coef_n_diff_gt_0.01", "coef_n_diff_gt_0.05",
           "full_pipeline_100splits", "m_100splits",
           "dev_full_n", "dev_full_male_pct"),
  Value = c(nrow(raw_val), val_n_missing,
            sum(imp_train$predictorMatrix["Outcome", ]),
            sum(imp_train$predictorMatrix[, "Outcome"]),
            imp_train$method["Outcome"],
            sum(imp_all$ignore),
            round(max(abs(cmp$Diff), na.rm = TRUE), 4),
            sum(abs(cmp$Diff) > 0.01, na.rm = TRUE),
            sum(abs(cmp$Diff) > 0.05, na.rm = TRUE),
            as.integer(FULL_PIPELINE_100SPLITS),
            M_100SPLITS,
            nrow(dev_full),
            round(100 * mean(male_flag, na.rm = TRUE), 2))
)
write.csv(check_summary, "results/v54_check_summary.csv",
          row.names = FALSE, quote = FALSE)
cat("\n===== v5.4 核对摘要 =====\n")
print(check_summary)

# 运行环境与关键分析口径记录
writeLines(c(
  "Pipeline version: v5.4.1",
  paste0("R version: ", R.version.string),
  paste0("mice version: ", as.character(packageVersion("mice"))),
  paste0("Development cohort n: ", nrow(raw)),
  paste0("Training n: ", nrow(raw_train)),
  paste0("Internal validation n: ", nrow(raw_val)),
  paste0("Training events: ", sum(raw_train$Outcome)),
  paste0("Validation events: ", sum(raw_val$Outcome)),
  "HCO3 column name normalized from HCO3_ to HCO3 after read.",
  "Imputation predictors exclude Outcome in training and validation.",
  "Primary model: Outcome ~ ALB + BUN + BE",
  "HCO3 model: Outcome ~ ALB + BUN + HCO3",
  "DD extension: Outcome ~ DD + ALB + BUN + BE",
  "Development baseline exports HCO3 mean and SD for supplementary-table generation.",
  "100-split analysis repeats the fixed prespecified model; it does not repeat variable selection."
), "results/run_manifest_v54.txt")

############################################################
# END OF R_analysis.R (v5.4.1)
############################################################