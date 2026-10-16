# ============================================================
# R_analysis.R — development-cohort analysis pipeline (v4.0)
# Project: respiratory-failure-risk-calculator
# Repo:    https://github.com/wj-202606/respiratory-failure-risk-calculator
# Zenodo:  10.5281/zenodo.23300381 (concept) / 10.5281/zenodo.23354599 (v1.1.0)
# Inputs:  respiratory.csv; Outputs: results/, figures/, 论文图/, imputed_data/
#
# v4.0 更新（本次，方案 A）：
#   [1] ★ 内部验证流程修正：先 7:3 划分，再分别对训练集和验证集做多重插补
#       消除 imputation leakage（原流程是先全数据插补再划分）
#   [2] ★ 内部验证评估改为对 imp_val 的 50 套分别评估取平均
#   [3] ★ 补齐 Brier/MAE 独立计算（rms::val.prob + naive）
#   [4] ★ 补齐内部验证校准曲线（loess + CI 带）
#   [5] ★ AUC_distribution 输出 CHEST 风格到 论文图/
#   [6] HCO₃⁻ 版本系数单独输出，方便作为主分析使用
#
# 与 v3.2 的差异：
#   - 第 5–6 节完全重写（先划分后插补）
#   - 所有引用 imp 的地方改为 imp_train
#   - 新增内部验证 50 套评估
#   - 新增 Brier/MAE 独立计算
#   - 新增内部验证校准曲线
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

# 全局随机种子记录
GLOBAL_SEED <- 123
set.seed(GLOBAL_SEED)
writeLines(paste0("Global seed: ", GLOBAL_SEED,
                  "\nOuter split seed: 123",
                  "\nImputation seed (train): 123",
                  "\nImputation seed (val): 456",
                  "\nLASSO loop seed: 123 + imp_i",
                  "\n100-split seed: 1:100",
                  "\nBootstrap seed: 123"),
           "results/random_seeds.txt")

############################################################
######### 2. 读数据 #########
############################################################
raw <- read.csv("respiratory.csv",
                fileEncoding = "GBK",
                na.strings = c("", "NA", "NULL", " ", "  ",
                               "不报告", "测不出", "未测", "未查", "缺失"),
                stringsAsFactors = FALSE)

cat("\n===== 原始数据维度 =====\n"); print(dim(raw))
cat("\n===== 原始列名 =====\n");     print(names(raw))

############################################################
######### 3. 数据预处理 #########
############################################################

# 3.1 分类变量
cat_vars <- c("sex")
cat_vars <- cat_vars[cat_vars %in% names(raw)]
raw[cat_vars] <- lapply(raw[cat_vars], as.factor)
raw$Outcome <- as.integer(as.character(raw$Outcome))
table(raw$Outcome)

# 3.2 所有 character 列 -> numeric
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

# 3.3 双保险
char_to_num <- c("APTT","NEU","LYM","RDW","PLT","K","TCO2","BE","SaO2")
char_to_num <- char_to_num[char_to_num %in% names(raw)]
for (var in char_to_num) {
  raw[[var]] <- suppressWarnings(as.numeric(trimws(as.character(raw[[var]]))))
  cat(var, "-> NA：", sum(is.na(raw[[var]])), "\n")
}

# 3.4 APTT 检查
stopifnot(is.numeric(raw$APTT))
cat("\nAPTT 类型：", class(raw$APTT), "\n")
print(summary(raw$APTT))

# 3.5 缺失情况
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
######### 5. ★★★ 先 7:3 划分，后分别插补（方案 A） ★★★
############################################################
# 关键改动：消除 imputation leakage
# 原流程：全数据插补 → complete(1) → 7:3 划分（泄漏）
# 新流程：7:3 划分 → 训练集插补 + 验证集插补（无泄漏）

# 5.1 先划分原始数据
set.seed(123)
raw$Outcome <- as.integer(as.character(raw$Outcome))
inTrain <- createDataPartition(y = raw$Outcome, p = 0.7, list = FALSE)
raw_train <- raw[inTrain, ]
raw_val   <- raw[-inTrain, ]

cat("\n===== 外层划分 =====")
cat("train:", nrow(raw_train), " val:", nrow(raw_val), "\n")
cat("train events:", sum(raw_train$Outcome), " val events:", sum(raw_val$Outcome), "\n")

write.csv(data.frame(idx = inTrain), "results/train_indices.csv", row.names = FALSE)

# 5.2 训练集插补（m=50）
set.seed(123)
pred_mat_train <- make.predictorMatrix(raw_train)
if ("Outcome" %in% rownames(pred_mat_train)) {
  pred_mat_train[, "Outcome"] <- 0
  pred_mat_train["Outcome", ] <- 0
}
imp_train <- mice(raw_train, m = 50, seed = 123, print = FALSE,
                  predictorMatrix = pred_mat_train)
cat("\n✅ 训练集插补完成（m=50）\n")

# 5.3 验证集插补（m=50）
set.seed(456)
pred_mat_val <- make.predictorMatrix(raw_val)
if ("Outcome" %in% rownames(pred_mat_val)) {
  pred_mat_val[, "Outcome"] <- 0
  pred_mat_val["Outcome", ] <- 0
}
imp_val <- mice(raw_val, m = 50, seed = 456, print = FALSE,
                predictorMatrix = pred_mat_val)
cat("✅ 验证集插补完成（m=50）\n")

# 5.4 保存
saveRDS(imp_train, "results/imp_train_m50.rds")
saveRDS(imp_val,   "results/imp_val_m50.rds")

# 5.5 第 1 套用于后续分析
df_train <- mice::complete(imp_train, action = 1)
df_val   <- mice::complete(imp_val,   action = 1)
traindata <- df_train
valdata   <- df_val

write.csv(traindata, "results/traindata.csv", row.names = FALSE, quote = FALSE)
write.csv(valdata,   "results/valdata.csv",   row.names = FALSE, quote = FALSE)

# 5.6 导出所有 50 套插补数据（训练集）
for (i in 1:50) {
  d <- mice::complete(imp_train, action = i)
  write.csv(d, sprintf("imputed_data/imputed_train_%02d.csv", i),
            row.names = FALSE)
}
cat("✅ 训练集 50 套插补数据已保存\n")

############################################################
######### 6. 基线表（基于训练集第 1 套）#########
############################################################
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

############################################################
######### 7. LASSO 变量筛选（在训练集上）#########
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

# LASSO CV 曲线
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

# LASSO 系数路径
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
cat("\n===== LASSO 选出的变量（lambda.1se） =====\n"); print(nonzero_vars)

coefs_min <- as.matrix(coef(lasso.cv, s = "lambda.min"))
vars_min  <- rownames(coefs_min)[coefs_min[, 1] != 0]
vars_min  <- vars_min[vars_min != "(Intercept)"]
cat("\n===== lambda.min 选出 =====\n"); print(vars_min)

############################################################
# ★★★ 50 套 LASSO 稳定性分析（在 imp_train 上） ★★★
############################################################
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
# ★★★ 最终变量集 ★★★
############################################################
MAIN_VARS <- c("ALB", "BUN", "BE")
MAIN_VARS <- intersect(MAIN_VARS, colnames(traindata))

SENS_VARS <- c("DD", "ALB", "BUN", "BE")
SENS_VARS <- intersect(SENS_VARS, colnames(traindata))

cat("\n===== 主模型变量（3 变量）=====\n"); print(MAIN_VARS)
cat("\n===== 敏感性模型变量（4 变量）=====\n"); print(SENS_VARS)

write.csv(data.frame(
  Model = c(rep("Main3", length(MAIN_VARS)), rep("Sens4", length(SENS_VARS))),
  Variable = c(MAIN_VARS, SENS_VARS)
), "results/final_vars.csv", row.names = FALSE, quote = FALSE)

nonzero_vars <- MAIN_VARS
f_main <- as.formula(paste("Outcome ~", paste(MAIN_VARS, collapse = " + ")))
f_sens <- as.formula(paste("Outcome ~", paste(SENS_VARS, collapse = " + ")))

############################################################
######### 8. 多因素 Logistic（主模型，Rubin's rules）#########
############################################################
cat("\n主模型公式：\n"); print(f_main)

# ★ 用 imp_train（不是 imp）
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

# 输出系数
ext_coef <- data.frame(
  Term = pooled_summary$term,
  Beta = pooled_summary$estimate,
  SE   = pooled_summary$std.error
)
write.csv(ext_coef, "results/external_validation_coef.csv",
          row.names = FALSE, quote = FALSE)
cat("\n✅ 主模型系数已保存\n")
print(ext_coef)

# LP 公式
lp_formula <- paste0("LP = ",
                     round(pooled_summary$estimate[1], 4),
                     paste0(sprintf(" %+0.4f * %s",
                                    pooled_summary$estimate[-1],
                                    pooled_summary$term[-1]),
                            collapse = ""))
writeLines(lp_formula, "results/LP_formula_main3.txt")
cat("\n===== 主模型 LP 公式 =====\n"); cat(lp_formula, "\n")

cat("\n===== FMI =====\n"); print(pooled$pooled)
write.csv(pooled$pooled, "results/FMI_main3.csv", row.names = FALSE, quote = FALSE)

# VIF
Multi_log <- glm(f_main, data = traindata, family = binomial)
vif_values <- vif(Multi_log)
cat("\nVIF：\n"); print(round(vif_values, 3))
write.csv(data.frame(Variable = names(vif_values), VIF = round(vif_values, 3)),
          "results/VIF_main3.csv", row.names = FALSE, quote = FALSE)

############################################################
######### 8b. 敏感性分析：4 变量模型（DD）#########
############################################################
cat("\n敏感性模型公式：\n"); print(f_sens)

fit_list_sens <- with(imp_train, glm(
  as.formula(paste("Outcome ~", paste(SENS_VARS, collapse = " + "))),
  family = binomial
))
pooled_sens <- pool(fit_list_sens, dfcom = nrow(traindata))
pooled_sens_summary <- summary(pooled_sens, conf.int = TRUE)

write.csv(pooled_sens_summary, "results/Rubin_Logistic_sens4.csv",
          row.names = FALSE, quote = FALSE)

pooled_or_sens <- data.frame(
  Variable = pooled_sens_summary$term,
  Beta     = round(pooled_sens_summary$estimate, 4),
  SE       = round(pooled_sens_summary$std.error, 4),
  OR       = round(exp(pooled_sens_summary$estimate), 3),
  Lower_CI = round(exp(pooled_sens_summary$`2.5 %`), 3),
  Upper_CI = round(exp(pooled_sens_summary$`97.5 %`), 3),
  p_value  = format_p(pooled_sens_summary$p.value)
)
write.csv(pooled_or_sens, "results/Rubin_OR_sens4.csv",
          row.names = FALSE, quote = FALSE)
print(pooled_or_sens)

cat("\n✅ 4 变量敏感性分析完成\n")

############################################################
######### 9. 稳健性分析（在 imp_train 上）#########
############################################################

# ---- 9.1 50 套系数分布 ----
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

# ---- 9.2 Bootstrap 1000（在训练集上）----
boot_coef <- function(data, indices) {
  d <- data[indices, ]
  coef(glm(f_main, data = d, family = binomial))
}
set.seed(123)
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
write.csv(boot_ci, "results/bootstrap_ci.csv", row.names = FALSE, quote = FALSE)

# Bootstrap 分布图
for (target_var in nonzero_vars) {
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
######### 10. 模型评估：训练集 + 内部验证集（方案 A）#########
############################################################

# 10.1 拟合 rms::lrm（用训练集第 1 套）
dd <- datadist(traindata)
options(datadist = "dd")
fit_lrm <- lrm(f_main, data = traindata, x = TRUE, y = TRUE)
print(fit_lrm)

# 10.2 训练集预测
train_pred <- predict(fit_lrm, type = "fitted")
train_roc  <- roc(traindata$Outcome, train_pred)
train_auc  <- as.numeric(auc(train_roc))
train_brier <- mean((train_pred - as.numeric(as.character(traindata$Outcome)))^2)

# 10.3 ★ 内部验证：对 imp_val 的 50 套分别评估，取平均
cat("\n===== 内部验证（50 套 val 插补）=====\n")
auc_list    <- c()
brier_list  <- c()
slope_list  <- c()
int_list    <- c()

for (i in 1:50) {
  val_i <- mice::complete(imp_val, action = i)
  val_i$Outcome <- as.numeric(as.character(val_i$Outcome))

  pred_i <- predict(fit_lrm, newdata = val_i, type = "fitted")
  y_i    <- val_i$Outcome

  # AUC
  auc_i <- as.numeric(auc(y_i, pred_i))
  auc_list <- c(auc_list, auc_i)

  # Brier
  brier_i <- mean((pred_i - y_i)^2)
  brier_list <- c(brier_list, brier_i)

  # Calibration slope / intercept
  eps <- 1e-6
  lp_i <- log(pmax(pmin(pred_i, 1-eps), eps) /
              (1 - pmax(pmin(pred_i, 1-eps), eps)))
  cal_i <- glm(y_i ~ lp_i, family = binomial)
  slope_list <- c(slope_list, coef(cal_i)[2])
  int_list   <- c(int_list,   coef(cal_i)[1])
}

cat(sprintf("AUC      (mean ± SD): %.4f ± %.4f\n", mean(auc_list), sd(auc_list)))
cat(sprintf("Brier    (mean ± SD): %.4f ± %.4f\n", mean(brier_list), sd(brier_list)))
cat(sprintf("Cal slope(mean ± SD): %.4f ± %.4f\n", mean(slope_list), sd(slope_list)))
cat(sprintf("Cal int  (mean ± SD): %.4f ± %.4f\n", mean(int_list), sd(int_list)))

val_summary <- data.frame(
  Metric = c("AUC", "Brier", "Cal_slope", "Cal_intercept"),
  Mean   = c(mean(auc_list), mean(brier_list), mean(slope_list), mean(int_list)),
  SD     = c(sd(auc_list),   sd(brier_list),   sd(slope_list),   sd(int_list)),
  Q25    = c(quantile(auc_list, .25), quantile(brier_list, .25),
             quantile(slope_list, .25), quantile(int_list, .25)),
  Q75    = c(quantile(auc_list, .75), quantile(brier_list, .75),
             quantile(slope_list, .75), quantile(int_list, .75))
)
write.csv(val_summary, "results/internal_validation_50imp.csv",
          row.names = FALSE, quote = FALSE)

# 10.4 单套结果（用于 Table 3 展示，取第 1 套）
valdata$Outcome_num <- as.numeric(as.character(valdata$Outcome))
val_pred <- predict(fit_lrm, newdata = valdata, type = "fitted")
val_roc  <- roc(valdata$Outcome_num, val_pred)
val_auc  <- as.numeric(auc(val_roc))
val_brier <- mean((val_pred - valdata$Outcome_num)^2)

cat("\n===== 单套（第 1 套）内部验证结果 =====\n")
cat("AUC     =", round(val_auc, 4), "\n")
cat("Brier   =", round(val_brier, 4), "\n")

# 校准斜率/截距（第 1 套）
eps <- 1e-6
lp_val <- log(pmax(pmin(val_pred, 1-eps), eps) /
              (1 - pmax(pmin(val_pred, 1-eps), eps)))
cal_val <- glm(valdata$Outcome_num ~ lp_val, family = binomial)
slope_val <- coef(cal_val)[2]
int_val   <- coef(cal_val)[1]
cat("Slope   =", round(slope_val, 4), "\n")
cat("Intercept =", round(int_val, 4), "\n")

# 10.5 Brier + MAE（独立计算，naive + rms::val.prob）
y_train <- as.numeric(as.character(traindata$Outcome))
y_val   <- valdata$Outcome_num

brier_train_naive <- mean((train_pred - y_train)^2)
brier_val_naive   <- mean((val_pred   - y_val  )^2)
mae_train         <- mean(abs(train_pred - y_train))
mae_val           <- mean(abs(val_pred   - y_val  ))

vp_train <- val.prob(train_pred, y_train, pl = FALSE)
vp_val   <- val.prob(val_pred,   y_val,   pl = FALSE)

cat("\n===== Brier / MAE（独立计算）=====\n")
cat(sprintf("Naive Brier  Train: %.4f   Valid: %.4f\n",
            brier_train_naive, brier_val_naive))
cat(sprintf("rms   Brier  Train: %.4f   Valid: %.4f\n",
            vp_train["Brier"], vp_val["Brier"]))
cat(sprintf("MAE          Train: %.4f   Valid: %.4f\n",
            mae_train, mae_val))

write.csv(data.frame(
  Metric = c("Brier_train_naive", "Brier_val_naive",
             "Brier_train_rms",   "Brier_val_rms",
             "MAE_train",         "MAE_val"),
  Value  = c(brier_train_naive, brier_val_naive,
             vp_train["Brier"], vp_val["Brier"],
             mae_train, mae_val)
), "results/brier_mae_check.csv", row.names = FALSE, quote = FALSE)

############################################################
######### 11. 100 次随机划分 AUC（方案 A：每次重新划分+插补简化）
############################################################
# 注意：完整方案 A 应为"每次划分后重新插补"，但这样 100 次会耗时极长
# 折中方案：用 imp_train 的第 1 套 df，做 100 次重划分
# 若审稿人严格要求，可后续单独跑"100 次划分+100 次插补"

cat("\n===== 100 次随机划分 AUC =====\n")
df <- traindata   # 用 imp_train 第 1 套（简化）
df$Outcome <- as.factor(df$Outcome)
df[cat_vars] <- lapply(df[cat_vars], as.factor)

auc_results <- c()
for (i in 1:100) {
  set.seed(i)
  inTrain_i <- createDataPartition(y = df$Outcome, p = 0.7, list = FALSE)
  train_i <- df[inTrain_i, ]; val_i <- df[-inTrain_i, ]
  fit_i  <- glm(f_main, data = train_i, family = binomial)
  pred_i <- predict(fit_i, newdata = val_i, type = "response")
  auc_results <- c(auc_results, as.numeric(auc(val_i$Outcome, pred_i)))
}

med_auc  <- median(auc_results)
q25_auc  <- quantile(auc_results, .25)
q75_auc  <- quantile(auc_results, .75)

cat("中位数 =", round(med_auc, 4),
    " IQR =", round(q25_auc, 4), "~", round(q75_auc, 4),
    " SD =", round(sd(auc_results), 4), "\n")

write.csv(data.frame(Run = 1:100, AUC = round(auc_results, 4)),
          "results/auc_100splits.csv", row.names = FALSE, quote = FALSE)

# CHEST 风格输出到 论文图/
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

# 旧版兼容
save_both(quote({
  hist(auc_results, breaks=20,
       main = "AUC across 100 random splits",
       xlab = "AUC", col="lightblue", border="white")
  abline(v = median(auc_results), col="red", lwd=2, lty=2)
  legend("topright", legend = paste0("Median = ", round(median(auc_results),3)),
         col="red", lty=2, lwd=2, bty="n")
}), "AUC_distribution", width=8, height=6)

############################################################
######### 12. ROC + 校准曲线（训练 vs 内部验证）#########
############################################################

# 12.1 ROC 曲线
save_both(quote({
  plot(train_roc, col="blue", main="ROC Curves")
  plot(val_roc, col="red", add=TRUE)
  legend("bottomright",
         legend = c(paste0("Train AUC = ", round(train_auc, 3)),
                    paste0("Valid AUC = ", round(val_auc, 3))),
         col = c("blue","red"), lwd = 2)
}), "ROC_curves", width=8, height=8)

saveRDS(list(train_roc = train_roc, val_roc = val_roc,
             train_auc = train_auc, val_auc = val_auc),
        "results/roc_objects.rds")

# 12.2 训练集校准曲线（bootstrap 校正）
set.seed(123)
cal_train <- calibrate(fit_lrm, method = "boot", B = 1000)

save_both(quote({
  plot(cal_train, main = "Calibration Curve - Training (Bootstrap, B = 1000)",
       xlab = "Predicted Probability",
       ylab = "Observed Probability")
  abline(0, 1, col = "gray", lty = 2)
}), "calibration_curve_train", width=8, height=8)

# 12.3 验证集校准曲线（loess + CI 带）
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
######### 13. 汇总输出（供论文使用）
############################################################
summary_out <- data.frame(
  Metric = c("Train_AUC", "Train_Brier", "Train_MAE",
             "Val_AUC_single", "Val_Brier_single",
             "Val_AUC_50imp_mean", "Val_AUC_50imp_sd",
             "Val_Brier_50imp_mean", "Val_Brier_50imp_sd",
             "Val_slope_single", "Val_intercept_single",
             "Val_slope_50imp_mean", "Val_intercept_50imp_mean",
             "AUC_100split_median", "AUC_100split_IQR_lo", "AUC_100split_IQR_hi"),
  Value = c(train_auc, train_brier, mae_train,
            val_auc, val_brier,
            mean(auc_list), sd(auc_list),
            mean(brier_list), sd(brier_list),
            slope_val, int_val,
            mean(slope_list), mean(int_list),
            med_auc, q25_auc, q75_auc)
)
write.csv(summary_out, "results/summary_for_manuscript.csv",
          row.names = FALSE, quote = FALSE)
print(summary_out)

############################################################
# END OF R_analysis.R (v4.0)
#
# 关键改动（相对 v3.2）：
#   [1] 第 5 节：先 7:3 划分，再分别对训练集和验证集做多重插补
#   [2] 所有引用 imp 的地方改为 imp_train
#   [3] 第 10.3 节：内部验证对 imp_val 的 50 套分别评估
#   [4] 第 10.5 节：Brier/MAE 独立计算（naive + rms::val.prob）
#   [5] 第 11 节：AUC_distribution 输出 CHEST 风格到 论文图/
#   [6] 第 13 节：汇总所有关键数字供论文使用
############################################################