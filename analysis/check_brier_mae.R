# ============================================================
# check_brier_mae.R
#
# Purpose:
#   Standalone reproducibility check for the Brier score and MAE
#   reported for the internal validation subset (Table 3 of the
#   manuscript).
#
#   Reads the pre-computed training and internal validation
#   subsets (results/traindata.csv, results/valdata.csv), refits
#   the primary 3-variable logistic model (ALB + BUN + BE), and
#   reports Brier score and MAE using both a naive formula and
#   rms::val.prob.
#
# Expected values (manuscript, Table 3):
#   Training Brier            = 0.0899
#   Internal validation Brier = 0.0916
#
#   Small numerical differences (<= 0.001) between this script
#   and the manuscript values may arise because the manuscript
#   values were obtained from the full analysis pipeline
#   (multiple imputation + LASSO + model fit) in a single
#   session, whereas this script refits the model on the
#   already-saved subsets.
#
# Inputs:
#   results/traindata.csv   (n = 1,421)
#   results/valdata.csv     (n = 607)
#
# Outputs:
#   Printed to console only (no files written).
#
# Repo:   https://github.com/wj-202606/respiratory-failure-risk-calculator
# Zenodo: 10.5281/zenodo.23300381 (concept) / 10.5281/zenodo.23354599 (v1.1.0)
# ============================================================

suppressPackageStartupMessages({
  library(rms)
})

## 0. Working directory: set R_WD environment variable, or edit below.
if (nzchar(Sys.getenv("R_WD"))) {
  setwd(Sys.getenv("R_WD"))
} else {
  # setwd("path/to/project")  # uncomment and edit if needed
}

## 1. Read pre-computed subsets
traindata <- read.csv("results/traindata.csv", stringsAsFactors = FALSE)
valdata   <- read.csv("results/valdata.csv",   stringsAsFactors = FALSE)

cat("traindata rows:", nrow(traindata), " valdata rows:", nrow(valdata), "\n")
cat("traindata events:", sum(as.numeric(traindata$Outcome) == 1), "\n")
cat("valdata events:   ", sum(as.numeric(valdata$Outcome)   == 1), "\n")

## 2. Type harmonisation
traindata$Outcome <- factor(traindata$Outcome, levels = c(0, 1))
valdata$Outcome   <- factor(valdata$Outcome,   levels = c(0, 1))
if ("sex" %in% names(traindata)) traindata$sex <- as.factor(traindata$sex)
if ("sex" %in% names(valdata))   valdata$sex   <- as.factor(valdata$sex)

## 3. Fit primary model (same formula as R_analysis.R)
f_main <- Outcome ~ ALB + BUN + BE

dd <- datadist(traindata); options(datadist = "dd")
fit_lrm <- lrm(f_main, data = traindata, x = TRUE, y = TRUE)

## 4. Predictions
train_pred <- predict(fit_lrm, type = "fitted")
val_pred   <- predict(fit_lrm, newdata = valdata, type = "fitted")

## 5. Brier + MAE (naive formula and rms::val.prob)
y_train <- as.numeric(as.character(traindata$Outcome))
y_val   <- as.numeric(as.character(valdata$Outcome))

brier_train_naive <- mean((train_pred - y_train)^2)
brier_val_naive   <- mean((val_pred   - y_val  )^2)
mae_train         <- mean(abs(train_pred - y_train))
mae_val           <- mean(abs(val_pred   - y_val  ))

vp_train <- val.prob(train_pred, y_train, pl = FALSE)
vp_val   <- val.prob(val_pred,   y_val,   pl = FALSE)

cat("\n===== Results =====\n")
cat(sprintf("Naive Brier  Train: %.4f   Valid: %.4f\n",
            brier_train_naive, brier_val_naive))
cat(sprintf("rms   Brier  Train: %.4f   Valid: %.4f\n",
            vp_train["Brier"], vp_val["Brier"]))
cat(sprintf("MAE          Train: %.4f   Valid: %.4f\n",
            mae_train, mae_val))

cat("\nManuscript reference values (Table 3):\n")
cat("  Training Brier            = 0.0899\n")
cat("  Internal validation Brier = 0.0916\n")
cat("\nSmall differences (<= 0.001) reflect the fact that the manuscript values\n")
cat("were obtained from the full pipeline in a single session, whereas this\n")
cat("script refits the model on the pre-computed subsets.\n")