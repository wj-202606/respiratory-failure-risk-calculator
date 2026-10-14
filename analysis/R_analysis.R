# ============================================================
# R_analysis.R — development-cohort analysis pipeline
# Project: respiratory-failure-risk-calculator
# Repo:    https://github.com/wj-202606/respiratory-failure-risk-calculator
# Zenodo:  10.5281/zenodo.23300381 (concept) / 10.5281/zenodo.23354599 (v1.1.0)
# Inputs:  respiratory.csv; Outputs: results/, figures/, imputed_data/
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
# setwd() - set your working directory manually, or set the R_WD environment variable
if (nzchar(Sys.getenv("R_WD"))) setwd(Sys.getenv("R_WD"))

if (.Platform$OS.type == "windows")
  windowsFonts(Times = windowsFont("Times New Roman"))
par(family = "Times")

dir.create("results",      showWarnings = FALSE)
dir.create("figures",      showWarnings = FALSE)
dir.create("imputed_data", showWarnings = FALSE)

# ★ 统一画图函数
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

# ★ 全局随机种子记录
GLOBAL_SEED <- 123
set.seed(GLOBAL_SEED)
writeLines(paste0("Global seed: ", GLOBAL_SEED,
                  "\nImputation seed: 123",
                  "\nLASSO loop seed: 123 + imp_i",
                  "\n100-split seed: 1:100",
                  "\nBootstrap seed: 123",
                  "\nOuter split seed: 123"),
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
######### 5. 多重插补 m=50 #########
############################################################
# ★ 显式 predictorMatrix，排除 Outcome 作为插补预测变量
set.seed(123)
pred_mat <- make.predictorMatrix(raw)
if ("Outcome" %in% rownames(pred_mat)) {
  pred_mat[, "Outcome"] <- 0      # Outcome 不作为预测变量
  pred_mat["Outcome", ] <- 0      # Outcome 本身不插补
}
imp <- mice(raw, m = 50, seed = 123, print = FALSE,
            predictorMatrix = pred_mat)

cat("\n===== 插补方法 =====\n")
print(imp$method)

complete_data <- mice::complete(imp, 1)
imp_method    <- imp$method

all_types <- sapply(complete_data, class)
cat("\n===== 全变量类型 =====\n")
print(all_types)

if (any(all_types == "character")) {
  stop("存在 character 类型变量：",
       paste(names(all_types)[all_types == "character"], collapse = ", "))
}
cat("✅ 无 character 类型变量\n")

issues <- c()
for (v in names(imp_method)) {
  if (imp_method[v] == "") next
  cls <- class(complete_data[[v]])[1]
  m   <- imp_method[v]
  if (cls == "numeric" && m %in% c("logreg", "polyreg", "polr")) {
    issues <- c(issues, paste0(v, " (", cls, " 用了 ", m, ")"))
  }
  if (cls %in% c("factor", "character") && m %in% c("pmm", "norm")) {
    issues <- c(issues, paste0(v, " (", cls, " 用了 ", m, ")"))
  }
}

if (length(issues) > 0) {
  cat("\n⚠️ 类型与方法不匹配：\n"); print(issues)
} else {
  cat("\n✅ 所有变量插补方法与类型匹配\n")
}

if (imp_method["Outcome"] == "" || is.na(imp_method["Outcome"])) {
  cat("✅ Outcome 未被插补\n")
} else {
  cat("⚠️ Outcome 被插补，方法：", imp_method["Outcome"], "\n")
}

imp_diag <- data.frame(
  Variable     = names(imp_method),
  Class        = sapply(complete_data[, names(imp_method), drop = FALSE], class),
  ImputeMethod = as.character(imp_method)
)
write.csv(imp_diag, "results/imp_diagnostics.csv",
          row.names = FALSE, quote = FALSE)
cat("✅ 插补诊断表已保存\n")

for (i in 1:50) {
  d <- mice::complete(imp, action = i)
  write.csv(d, sprintf("imputed_data/imputed_%02d.csv", i),
            row.names = FALSE)
}
cat("✅ 50 套插补数据已保存\n")

saveRDS(imp, "results/imp_m50.rds")

############################################################
######### 6. 划分训练集 / 验证集（7:3）#########
############################################################
df <- mice::complete(imp, action = 1)
df$Outcome <- factor(df$Outcome, levels = c(0, 1))
df[cat_vars] <- lapply(df[cat_vars], as.factor)

set.seed(123)
inTrain <- createDataPartition(y = df$Outcome, p = 0.7, list = FALSE)
traindata <- df[inTrain, ]
valdata   <- df[-inTrain, ]
traindata[cat_vars] <- lapply(traindata[cat_vars], as.factor)
valdata[cat_vars]   <- lapply(valdata[cat_vars],   as.factor)

cat("\n训练集：", nrow(traindata), " 验证集：", nrow(valdata), "\n")
write.csv(traindata, "results/traindata.csv", row.names = FALSE, quote = FALSE)
write.csv(valdata,   "results/valdata.csv",   row.names = FALSE, quote = FALSE)

write.csv(data.frame(idx = inTrain), "results/train_indices.csv",
          row.names = FALSE)

saveRDS(list(traindata = traindata, valdata = valdata, inTrain = inTrain),
        "results/split.rds")

############################################################
######### 7. 基线表 #########
############################################################
catVars  <- cat_vars
contVars <- setdiff(colnames(df), c("Outcome", catVars))
myVars   <- c(catVars, contVars)

noncenter <- c()
for (var in contVars) {
  if (is.numeric(df[[var]])) {
    sk <- skewness(df[[var]], na.rm = TRUE)
    if (abs(sk) > 1) noncenter <- c(noncenter, var)
  }
}
cat("\n非正态变量（|偏度|>1）：\n"); print(noncenter)

t1 <- CreateTableOne(vars = myVars, factorVars = catVars,
                     data = df, strata = "Outcome", addOverall = TRUE)
write.csv(print(t1, showAllLevels=TRUE, catDigits=2, contDigits=2, pDigits=3,
                nonnormal=noncenter, smd=TRUE, quote=FALSE, noSpaces=TRUE,
                printToggle=FALSE),
          "results/table1_overall.csv", quote = FALSE)

t2 <- CreateTableOne(vars = myVars, factorVars = catVars,
                     data = traindata, strata = "Outcome", addOverall = TRUE)
write.csv(print(t2, showAllLevels=TRUE, catDigits=2, contDigits=2, pDigits=3,
                nonnormal=noncenter, smd=TRUE, quote=FALSE, noSpaces=TRUE,
                printToggle=FALSE),
          "results/table1_train.csv", quote = FALSE)

############################################################
######### 8. LASSO 变量筛选 #########
############################################################
# ★ z-score 标准化（替代 min-max）
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
cat("✅ 已保存 LASSO_cv_curve.pdf / .png\n")

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
cat("✅ 已保存 LASSO_coef_path.pdf / .png\n")

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
# ★★★ 50 套 LASSO 稳定性分析 ★★★
############################################################
lasso_vars_by_imp <- list()
for (imp_i in 1:50) {
  d_i <- mice::complete(imp, action = imp_i)
  d_i$Outcome <- as.numeric(as.character(d_i$Outcome))
  train_i <- d_i[inTrain, ]

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
  Selected_Count = sapply(all_vars, function(v) sum(sapply(lasso_vars_by_imp, function(x) v %in% x)))
)
freq_table$Selected_Pct <- round(100 * freq_table$Selected_Count / 50, 1)
freq_table <- freq_table[order(-freq_table$Selected_Count), ]
print(freq_table)
write.csv(freq_table, "results/LASSO_vars_frequency.csv",
          row.names = FALSE, quote = FALSE)
cat("✅ 50 套 LASSO 稳定性分析完成\n")

############################################################
# ★★★ 最终变量集 ★★★
# 主模型：3 变量（ALB + BUN + BE），与 MIMIC-IV 外部验证对齐
# 敏感性模型：4 变量（+ DD）
############################################################
MAIN_VARS <- c("ALB", "BUN", "BE")
MAIN_VARS <- intersect(MAIN_VARS, colnames(traindata))

SENS_VARS <- c("DD", "ALB", "BUN", "BE")
SENS_VARS <- intersect(SENS_VARS, colnames(traindata))

cat("\n===== 主模型变量（3 变量，用于外部验证）=====\n"); print(MAIN_VARS)
cat("\n===== 敏感性模型变量（4 变量）=====\n"); print(SENS_VARS)

write.csv(data.frame(
  Model = c(rep("Main3", length(MAIN_VARS)), rep("Sens4", length(SENS_VARS))),
  Variable = c(MAIN_VARS, SENS_VARS)
), "results/final_vars.csv", row.names = FALSE, quote = FALSE)

nonzero_vars <- MAIN_VARS
f_main <- as.formula(paste("Outcome ~", paste(MAIN_VARS, collapse = " + ")))
f_sens <- as.formula(paste("Outcome ~", paste(SENS_VARS, collapse = " + ")))

############################################################
######### 9. 多因素 Logistic（主模型 3 变量）#########
############################################################
cat("\n主模型公式：\n"); print(f_main)

# ★ 修复：在 with() 内部构造公式，避免环境错位
fit_list <- with(imp, glm(
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

# ★ 输出外部验证系数
ext_coef <- data.frame(
  Term = pooled_summary$term,
  Beta = pooled_summary$estimate,
  SE   = pooled_summary$std.error
)
write.csv(ext_coef, "results/external_validation_coef.csv",
          row.names = FALSE, quote = FALSE)
cat("\n✅ 外部验证系数已保存：results/external_validation_coef.csv\n")
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

Multi_log <- glm(f_main, data = traindata, family = binomial)
vif_values <- vif(Multi_log)
cat("\nVIF：\n"); print(round(vif_values, 3))
write.csv(data.frame(Variable = names(vif_values), VIF = round(vif_values, 3)),
          "results/VIF_main3.csv", row.names = FALSE, quote = FALSE)

############################################################
######### 9b. 敏感性分析：4 变量模型 #########
############################################################
cat("\n敏感性模型公式：\n"); print(f_sens)

# ★ 修复：同样在 with() 内部构造公式
fit_list_sens <- with(imp, glm(
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
######### 10. 稳健性分析 #########
############################################################

# ---- 10.1 50 套系数分布 ----
coef_list <- lapply(1:50, function(i) {
  d <- mice::complete(imp, action = i)
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

# ---- 10.2 Bootstrap 1000 ----
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

# ★ Bootstrap 分布图（函数式绘制，修复图例重叠）
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

# ---- 10.3 100 次随机划分 AUC ----
auc_results <- c()
for (i in 1:100) {
  set.seed(i)
  inTrain_i <- createDataPartition(y = df$Outcome, p = 0.7, list = FALSE)
  train_i <- df[inTrain_i, ]; val_i <- df[-inTrain_i, ]
  fit_i  <- glm(f_main, data = train_i, family = binomial)
  pred_i <- predict(fit_i, newdata = val_i, type = "response")
  auc_results <- c(auc_results, as.numeric(auc(val_i$Outcome, pred_i)))
}
cat("\n===== 100 次划分验证集 AUC =====\n")
cat("中位数 =", round(median(auc_results), 4),
    " IQR =", round(quantile(auc_results, .25), 4),
    "~", round(quantile(auc_results, .75), 4),
    " SD =", round(sd(auc_results), 4), "\n")

write.csv(data.frame(Run = 1:100, AUC = round(auc_results, 4)),
          "results/auc_100splits.csv", row.names = FALSE, quote = FALSE)

save_both(quote({
  hist(auc_results, breaks=20,
       main = "AUC across 100 random splits",
       xlab = "AUC", col="lightblue", border="white")
  abline(v = median(auc_results), col="red", lwd=2, lty=2)
  legend("topright", legend = paste0("Median = ", round(median(auc_results),3)),
         col="red", lty=2, lwd=2, bty="n")
}), "AUC_distribution", width=8, height=6)

############################################################
######### 11. ROC 曲线（训练 vs 验证）#########
############################################################
dd <- datadist(traindata)
options(datadist = "dd")
fit_lrm <- lrm(f_main, data = traindata, x = TRUE, y = TRUE)

train_pred <- predict(fit_lrm, type = "fitted")
train_roc  <- roc(traindata$Outcome, train_pred)
train_auc  <- as.numeric(auc(train_roc))

val_pred <- predict(fit_lrm, newdata = valdata, type = "fitted")
val_roc  <- roc(valdata$Outcome, val_pred)
val_auc  <- as.numeric(auc(val_roc))

cat("\n===== ROC =====\n")
cat("训练集 AUC =", round(train_auc, 4), "\n")
cat("验证集 AUC =", round(val_auc, 4), "\n")

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

############################################################
######### 12. 校准曲线（训练集 + 验证集）#########
############################################################
# ★ 训练集校准曲线（bootstrap 校正）
set.seed(123)
cal_train <- calibrate(fit_lrm, method = "boot", B = 1000)

save_both(quote({
  plot(cal_train, main = "Calibration Curve - Training (Bootstrap, B = 1000)",
       xlab = "Predicted Probability",
       ylab = "Observed Probability")
  abline(0, 1, col = "gray", lty = 2)
}), "calibration_curve_train", width=8, height=8)

# ★ 验证集校准曲线（loess + CI 带）
val_pred_prob <- predict(fit_lrm, newdata = valdata, type = "fitted")
val_y_num     <- as.numeric(as.character(valdata$Outcome))

save_both(quote({
  cal_df <- data.frame(p = val_pred_prob, y = val_y_num)
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
# END OF R_analysis.R
#
# Note on scope:
#   This script covers data preprocessing, multiple imputation
#   (m = 50), LASSO variable selection, fitting of the primary
#   (BE) and sensitivity (HCO3-) logistic models, internal
#   validation (7:3 split, seed = 123), ROC curves, and
#   calibration curves.
#
#   Brier score and MAE for the internal validation subset were
#   computed interactively at the end of the analysis session
#   (values reported in Table 3 of the manuscript: training
#   Brier 0.0899; internal validation Brier 0.0916). External
#   validation, recalibration, bootstrap out-of-bag validation,
#   and decision curve analysis were implemented separately in
#   Python (see analysis/run_model.py, run_model_hco3.py, and
#   recal_bootstrap.py).
############################################################