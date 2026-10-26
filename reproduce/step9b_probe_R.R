# step9b_probe_R.R  探针：只读结构，不做分析
proj <- "C:/Users/24339/Desktop/6.30呼吸衰竭/6.30呼衰/1.R数据预处理+变量筛选9.21"
setwd(proj)

cat("== valdata.csv ==\n")
vd <- read.csv("results/valdata.csv", fileEncoding = "UTF-8")
cat("dim:", dim(vd), "\n")
cat("cols:", paste(names(vd), collapse=", "), "\n")
cat("na ALB/BUN/BE/Outcome:", sum(is.na(vd$ALB)), sum(is.na(vd$BUN)),
    sum(is.na(vd$BE)), sum(is.na(vd$Outcome)), "\n\n")

cat("== split.rds ==\n")
if (file.exists("results/split.rds")) {
  sp <- readRDS("results/split.rds")
  cat("class:", class(sp), "\n")
  cat("names/len:", if (!is.null(names(sp))) paste(names(sp), collapse=", ") else length(sp), "\n")
  if (is.list(sp) && !is.null(sp$inTrain)) cat("inTrain length:", length(sp$inTrain),
                                              " sum:", sum(sp$inTrain), "\n")
} else cat("split.rds 不存在\n")
cat("\n")

cat("== imp_val_m50.rds ==\n")
if (file.exists("results/imp_val_m50.rds")) {
  im <- readRDS("results/imp_val_m50.rds")
  cat("class:", class(im), "\n")
  if (inherits(im, "mids")) {
    cat("mids: m =", im$m, " nrow =", nrow(im$data), "\n")
    cat("vars:", paste(names(im$data), collapse=", "), "\n")
  } else if (is.list(im)) {
    cat("list length:", length(im), "\n")
    cat("first element class:", class(im[[1]]), "\n")
    if (is.data.frame(im[[1]])) cat("first df dim:", dim(im[[1]]),
                                    " cols:", paste(names(im[[1]]), collapse=", "), "\n")
  } else cat("未知类型\n")
} else cat("imp_val_m50.rds 不存在\n")
cat("\n")

cat("== imp_train_m50.rds（若存在，供核对）==\n")
if (file.exists("results/imp_train_m50.rds")) {
  it <- readRDS("results/imp_train_m50.rds")
  cat("class:", class(it), "\n")
  if (inherits(it, "mids")) cat("mids: m =", it$m, " nrow =", nrow(it$data), "\n")
  else if (is.list(it)) cat("list length:", length(it), " first class:", class(it[[1]]), "\n")
} else cat("imp_train_m50.rds 不存在\n")