# Portable Random Forest pipeline for manuscript sharing

suppressPackageStartupMessages({
  library(caret)
  library(pROC)
  library(dplyr)
  library(readxl)
})

# ---- Config ----
# Optional CLI overrides:
#   Rscript RF2_cleaned.R --input=data/Data.xlsx --sheet=1 --output=results/RandomForest_Model_Summary.csv
args <- commandArgs(trailingOnly = TRUE)
arg_list <- list()
for (a in args) {
  if (grepl("^--[^=]+=", a)) {
    kv <- strsplit(sub("^--", "", a), "=", fixed = TRUE)[[1]]
    arg_list[[kv[1]]] <- kv[2]
  }
}

input_path <- if (!is.null(arg_list$input)) arg_list$input else file.path("data", "Data.xlsx")
sheet_name <- if (!is.null(arg_list$sheet)) arg_list$sheet else 1
output_path <- if (!is.null(arg_list$output)) arg_list$output else file.path("results", "RandomForest_Model_Summary.csv")
train_label <- if (!is.null(arg_list$train_label)) tolower(trimws(arg_list$train_label)) else "training"
test_label <- if (!is.null(arg_list$test_label)) tolower(trimws(arg_list$test_label)) else "test"

if (!file.exists(input_path)) {
  stop(
    paste0(
      "Input file not found: ", input_path, "\n",
      "Provide --input=... or place your workbook at data/Data.xlsx"
    )
  )
}

output_dir <- dirname(output_path)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

data <- read_excel(input_path, sheet = sheet_name)
names(data) <- make.names(names(data), unique = TRUE)

# ---- Data preparation ----
required_raw <- c(
  "Patient",
  "Training or Test",
  "Sex",
  "Newly Diagnosed",
  "Trisomy 7",
  "Monosomy 10",
  "Sample Recurrence",
  "Age at surgery",
  "Tumor Volume",
  "Flair Volume",
  "Resection Cavity Volume",
  "Extent of Resection",
  "Tumor Mutation Burden",
  "Proliferation Index",
  "Methylation Score",
  "TERT VAF",
  "PTEN VAF",
  "TP53 VAF",
  "AI-infiltration"
)
required_columns <- make.names(required_raw)

missing_columns <- setdiff(required_columns, names(data))
if (length(missing_columns) > 0) {
  stop(paste("Missing required columns:", paste(missing_columns, collapse = ", ")))
}

categorical_vars <- make.names(c("Sex", "Newly Diagnosed", "Trisomy 7", "Monosomy 10"))
split_var <- make.names("Training or Test")
response_var <- make.names("Sample Recurrence")
predictors <- make.names(c(
  "Sex", "Newly Diagnosed", "Trisomy 7", "Monosomy 10",
  "Age at surgery", "Tumor Volume", "Flair Volume", "Resection Cavity Volume", "Extent of Resection",
  "Tumor Mutation Burden", "Proliferation Index", "Methylation Score", "TERT VAF",
  "PTEN VAF", "TP53 VAF", "AI-infiltration"
))

data <- data %>%
  mutate(
    Patient = suppressWarnings(as.numeric(as.character(Patient))),
    .split = trimws(tolower(as.character(.data[[split_var]])))
  )

train_data <- data %>%
  filter(.split == train_label)

test_data <- data %>%
  filter(.split == test_label)

if (nrow(train_data) == 0 || nrow(test_data) == 0) {
  stop(
    paste0(
      "No rows found in one or both split groups. ",
      "Check 'Training or Test' values or set --train_label and --test_label. ",
      "Current labels: train='", train_label, "', test='", test_label, "'."
    )
  )
}

cat("Training samples:", nrow(train_data), "\n")
cat("Test samples:", nrow(test_data), "\n")

train_data[categorical_vars] <- lapply(train_data[categorical_vars], as.factor)
test_data[categorical_vars] <- lapply(test_data[categorical_vars], as.factor)

formula <- as.formula(paste(response_var, "~", paste(predictors, collapse = " + ")))

train_data[[response_var]] <- factor(as.character(train_data[[response_var]]), levels = c("0", "1"), labels = c("No", "Yes"))
test_data[[response_var]] <- factor(as.character(test_data[[response_var]]), levels = c("0", "1"), labels = c("No", "Yes"))
cat("Updated response variable levels:", levels(train_data[[response_var]]), "\n")

# ---- Model training ----
cv_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  sampling = "up",
  savePredictions = TRUE
)

set.seed(123)
rf_model <- caret::train(
  formula,
  data = train_data,
  method = "rf",
  tuneLength = 10,
  trControl = cv_control,
  metric = "ROC"
)

fold_aucs <- rf_model$resample$ROC
average_auc <- mean(fold_aucs)
auc_sd <- sd(fold_aucs)
cat("Training AUC (10-fold CV):", average_auc, "±", auc_sd, "\n")

cv_preds <- rf_model$pred
cv_cm <- confusionMatrix(cv_preds$pred, cv_preds$obs, positive = "Yes")

test_probs <- predict(rf_model, newdata = test_data, type = "prob")
test_roc <- roc(test_data[[response_var]], test_probs$Yes, levels = c("No", "Yes"))
test_auc <- auc(test_roc)
cat("Test Set AUC:", round(test_auc, 4), "\n")

test_cm <- confusionMatrix(
  data = predict(rf_model, test_data),
  reference = test_data[[response_var]],
  positive = "Yes"
)
cat("Test Set Confusion Matrix:\n")
print(test_cm)

# ---- Variable importance ----
rf_importance <- varImp(rf_model, scale = FALSE)$importance
rf_importance$Metric <- rownames(rf_importance)
rf_importance$Value <- rf_importance$Overall
rf_importance <- rf_importance[, c("Metric", "Value")]

# ---- Metrics summary ----
summary_df <- data.frame(
  Metric = c(
    "Train AUC", "Train AUC SD",
    "Train Accuracy", "Train Kappa",
    "Train Sensitivity", "Train Specificity",
    "Train Pos Pred Value", "Train Neg Pred Value",
    "Train Prevalence", "Train Detection Rate",
    "Train Detection Prevalence", "Train Balanced Accuracy",
    "Train McNemar's Test P-Value",
    "Test AUC", "Test Accuracy", "Test Kappa",
    "Test Precision", "Test Recall", "Test F1-Score",
    "Test Sensitivity", "Test Specificity",
    "Test Pos Pred Value", "Test Neg Pred Value",
    "Test Prevalence", "Test Detection Rate",
    "Test Detection Prevalence", "Test Balanced Accuracy",
    "Test McNemar's Test P-Value"
  ),
  Value = c(
    average_auc, auc_sd,
    cv_cm$overall["Accuracy"], cv_cm$overall["Kappa"],
    cv_cm$byClass["Sensitivity"], cv_cm$byClass["Specificity"],
    cv_cm$byClass["Pos Pred Value"], cv_cm$byClass["Neg Pred Value"],
    cv_cm$byClass["Prevalence"], cv_cm$byClass["Detection Rate"],
    cv_cm$byClass["Detection Prevalence"], cv_cm$byClass["Balanced Accuracy"],
    cv_cm$overall["McnemarPValue"],
    test_auc, test_cm$overall["Accuracy"], test_cm$overall["Kappa"],
    test_cm$byClass["Precision"], test_cm$byClass["Recall"], test_cm$byClass["F1"],
    test_cm$byClass["Sensitivity"], test_cm$byClass["Specificity"],
    test_cm$byClass["Pos Pred Value"], test_cm$byClass["Neg Pred Value"],
    test_cm$byClass["Prevalence"], test_cm$byClass["Detection Rate"],
    test_cm$byClass["Detection Prevalence"], test_cm$byClass["Balanced Accuracy"],
    test_cm$overall["McnemarPValue"]
  )
)

combined_output <- rbind(summary_df, rf_importance)

set.seed(123)
test_auc_ci <- ci.auc(test_roc, boot.n = 2000, conf.level = 0.95, progress = "none")
cat("95% CI for Test Set AUC (bootstrapped):", round(test_auc_ci[1], 4), "-", round(test_auc_ci[3], 4), "\n")

cv_precision <- cv_cm$byClass["Precision"]
cv_recall <- cv_cm$byClass["Recall"]
cv_f1 <- cv_cm$byClass["F1"]
cat("\nCross-Validated Training Set Metrics:\n")
cat("Precision:", round(cv_precision, 4), "\n")
cat("Recall   :", round(cv_recall, 4), "\n")
cat("F1 Score :", round(cv_f1, 4), "\n")

write.csv(combined_output, output_path, row.names = FALSE)
cat("Training and test summary (including variable importance) saved to:", output_path, "\n")
