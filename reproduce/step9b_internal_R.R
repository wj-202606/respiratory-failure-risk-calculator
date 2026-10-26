# step9b_internal_R.R  内部验证口径定案（608）
proj <- "C:/Users/24339/Desktop/6.30呼吸衰竭/6.30呼衰/1.R数据预处理+变量筛选9.21"
setwd(proj)
suppressPackageStartupMessages({library(mice); library(pROC)})

vd <- read.csv("results/valdata.csv", fileEncoding="UTF-8")
stopifnot(nrow(vd)==608, sum(vd$Outcome==1)==72)
im <- readRDS("results/imp_val_m50.rds")
stopifnot(inherits(im,"mids"), im$m==50, nrow(im$data)==608)
cat("[口径校验] valdata 608/72；imp_val mids m=50 nrow=608 —— OK\n\n")

# 冻结系数（论文 Methods / Python 外部验证一致）
lp_frozen <- function(d) 1.7825 - 0.1227*d$ALB + 0.0343*d$BUN - 0.0585*d$BE

# R_analysis.R 的 fit 系数（若 Rubin_OR_main3.csv 存在则读，否则尝试 fit 对象）
coef_fit <- NULL
if (file.exists("results/Rubin_OR_main3.csv")) {
  ro <- read.csv("results/Rubin_OR_main3.csv")
  cat("Rubin_OR_main3.csv:\n"); print(ro)
  # 取 Beta 列，按 Variable 顺序：(Intercept), ALB, BUN, BE
  if (all(c("Variable","Beta") %in% names(ro))) {
    b <- setNames(ro$Beta, ro$Variable)
    coef_fit <- c(int=b[["(Intercept)"]], ALB=b[["ALB"]], BUN=b[["BUN"]], BE=b[["BE"]])
    cat("从 Rubin_OR_main3.csv 读出的系数：", paste(names(coef_fit), round(coef_fit,4), sep="=", collapse=", "), "\n\n")
  }
}
lp_fit <- function(d) {
  if (is.null(coef_fit)) return(NULL)
  coef_fit["int"] + coef_fit["ALB"]*d$ALB + coef_fit["BUN"]*d$BUN + coef_fit["BE"]*d$BE
}

metrics <- function(y,p){
  auc <- as.numeric(pROC::auc(pROC::roc(y,p,quiet=TRUE,direction="<")))
  brier <- mean((p-y)^2)
  eps<-1e-6; pc<-pmin(pmax(p,eps),1-eps); lp<-log(pc/(1-pc))
  f<-glm(y~lp,family=binomial())
  c(AUC=auc, Brier=brier, slope=unname(coef(f)[2]), intercept=unname(coef(f)[1]))
}

run50 <- function(lpfun, tag){
  R <- matrix(NA,50,4,dimnames=list(NULL,c("AUC","Brier","slope","intercept")))
  for(i in 1:50){
    d <- mice::complete(im,i); y <- as.integer(d$Outcome)
    lp <- lpfun(d); if(is.null(lp)) return(NULL)
    p <- 1/(1+exp(-lp)); R[i,] <- metrics(y,p)
  }
  cat(sprintf("\n== %s ==\n", tag))
  cat(sprintf("第1套: AUC %.4f  Brier %.4f  slope %.4f  int %.4f\n",
              R[1,"AUC"],R[1,"Brier"],R[1,"slope"],R[1,"intercept"]))
  cat(sprintf("50套均值: AUC %.4f±%.4f  Brier %.4f±%.4f  slope %.4f±%.4f  int %.4f±%.4f\n",
              mean(R[,"AUC"]),sd(R[,"AUC"]), mean(R[,"Brier"]),sd(R[,"Brier"]),
              mean(R[,"slope"]),sd(R[,"slope"]), mean(R[,"intercept"]),sd(R[,"intercept"])))
  invisible(R)
}

R_frozen <- run50(lp_frozen, "冻结系数 LP=1.7825...")
R_fit    <- run50(lp_fit,    "Rubin_OR_main3.csv 系数（R_analysis.R fit）")

cat("\n== FROZEN internal_validation_50imp.csv ==\n")
iv <- read.csv("results/FROZEN_v1/internal_validation_50imp.csv")
print(iv)
cat("\n判定：看 R_fit 的 50 套均值是否与 FROZEN 的 0.7181/0.0992/0.7439/-0.4352 一致；\n")
cat("若一致 → FROZEN 是 R_analysis.R fit 口径，正文单次 0.712 应改用该口径第 1 套；\n")
cat("若 R_frozen 更接近 → FROZEN 是冻结系数口径，正文单次应改用冻结系数第 1 套。\n")