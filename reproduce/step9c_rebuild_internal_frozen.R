# step9c_rebuild_internal_frozen.R
# 用 608 + 未中心化口径重建 internal_validation_50imp.csv
proj <- "C:/Users/24339/Desktop/6.30呼吸衰竭/6.30呼衰/1.R数据预处理+变量筛选9.21"
setwd(proj)
suppressPackageStartupMessages({library(mice); library(pROC)})
fz <- file.path("results","FROZEN_v1","internal_validation_50imp.csv")
if (file.exists(fz)) file.copy(fz, paste0(fz, ".bak_", format(Sys.Date(),"%Y%m%d")), overwrite=TRUE)

im <- readRDS("results/imp_val_m50.rds"); stopifnot(inherits(im,"mids"), im$m==50, nrow(im$data)==608)
lp_fun <- function(d) 1.7825 - 0.1227*d$ALB + 0.0343*d$BUN - 0.0585*d$BE
metrics <- function(y,p){
  auc <- as.numeric(pROC::auc(pROC::roc(y,p,quiet=TRUE,direction="<")))
  brier <- mean((p-y)^2)
  eps<-1e-6; pc<-pmin(pmax(p,eps),1-eps); lp<-log(pc/(1-pc))
  f<-glm(y~lp,family=binomial())   # 未中心化
  c(AUC=auc, Brier=brier, slope=unname(coef(f)[2]), intercept=unname(coef(f)[1]))
}
R <- matrix(NA,50,4,dimnames=list(NULL,c("AUC","Brier","Cal_slope","Cal_intercept")))
for(i in 1:50){ d<-mice::complete(im,i); y<-as.integer(d$Outcome); p<-1/(1+exp(-lp_fun(d))); R[i,]<-metrics(y,p) }
out <- data.frame(Metric=c("AUC","Brier","Cal_slope","Cal_intercept"),
                  Mean=colMeans(R), SD=apply(R,2,sd),
                  Q25=apply(R,2,quantile,.25), Q75=apply(R,2,quantile,.75))
print(out)
write.csv(out, fz, row.names=FALSE, quote=FALSE)
cat("\n已重建：", fz, "\n备份：", paste0(fz,".bak_",format(Sys.Date(),"%Y%m%d")), "\n")
cat("第1套单次：", paste(round(R[1,],4), collapse=" / "), "\n")