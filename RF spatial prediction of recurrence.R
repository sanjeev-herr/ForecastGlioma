# Spatial / recurrence RF models (<5 mm, <10 mm) — portable version
# Same workflow as CellularityFromRecurrence2.R: train on patients 1–60 with Recurred==1; test on 61–80.

suppressPackageStartupMessages({
  library(caret)
  library(pROC)
  library(dplyr)
  library(ggplot2)
  library(viridis)
  library(readxl)
})

# ---- Config ----
# Rscript CellularityFromRecurrence2_cleaned.R --input=data/Data.xlsx --sheet=1 --results_dir=results
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
results_dir <- if (!is.null(arg_list$results_dir)) arg_list$results_dir else "results"
# Column used for partial dependence plot (matches Data.xlsx; original used FastSRH)
pdp_raw <- if (!is.null(arg_list$pdp_var)) arg_list$pdp_var else "AI-infiltration"

if (!file.exists(input_path)) {
  stop(paste0("Input file not found: ", input_path, "\nProvide --input=... or place data at data/Data.xlsx"))
}
if (!dir.exists(results_dir)) {
  dir.create(results_dir, recursive = TRUE)
}

data <- read_excel(input_path, sheet = sheet_name)
names(data) <- make.names(names(data), unique = TRUE)

# Column names after make.names (Data.xlsx / manuscript-style headers)
response_5mm <- make.names("Recurrence less than 5 mm")
response_10mm <- make.names("Recurrence Less than 10 mm")
recurred_col <- make.names("Recurred")
pdp_col <- make.names(pdp_raw)

required_raw <- c(
  "Patient",
  "Recurred",
  "Recurrence less than 5 mm",
  "Recurrence Less than 10 mm",
  "Sex",
  "Newly Diagnosed",
  "Trisomy 7",
  "Monosomy 10",
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
  pdp_raw
)
required_columns <- make.names(required_raw)
missing_columns <- setdiff(required_columns, names(data))
if (length(missing_columns) > 0) {
  stop(paste("Missing required columns:", paste(missing_columns, collapse = ", ")))
}

categorical_vars <- make.names(c("Sex", "Newly Diagnosed", "Trisomy 7", "Monosomy 10"))
predictors <- c(
  categorical_vars,
  make.names(c(
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
    pdp_raw
  ))
)
eor_col <- make.names("Extent of Resection")

# Match original split logic
data <- data %>%
  mutate(Patient = suppressWarnings(as.numeric(as.character(Patient))))
data[[recurred_col]] <- suppressWarnings(as.numeric(as.character(data[[recurred_col]])))

train_data <- data %>%
  filter(!is.na(Patient), Patient >= 1 & Patient <= 60, .data[[recurred_col]] == 1)

test_data <- data %>%
  filter(!is.na(Patient), Patient >= 61 & Patient <= 80)

if (nrow(train_data) == 0) {
  stop("Training set is empty. Need rows with Patient in 1-60 and Recurred == 1.")
}
if (nrow(test_data) == 0) {
  stop("Test set is empty. Need rows with Patient in 61-80.")
}

cat("Training samples (1-60, Recurred==1):", nrow(train_data), "\n")
cat("Test samples (61-80):", nrow(test_data), "\n")

# Median impute Extent of Resection (train statistics only)
median_eor <- median(train_data[[eor_col]], na.rm = TRUE)
train_data[[eor_col]][is.na(train_data[[eor_col]])] <- median_eor
test_data[[eor_col]][is.na(test_data[[eor_col]])] <- median_eor

run_rf_model <- function(train_data, test_data, response_var, output_prefix, pdp_col, results_dir) {
  cat("\n===============================\n")
  cat("Running model for:", response_var, "\n")
  cat("===============================\n")

  train_data[[response_var]] <- factor(
    as.character(train_data[[response_var]]),
    levels = c("0", "1"),
    labels = c("No", "Yes")
  )
  test_data[[response_var]] <- factor(
    as.character(test_data[[response_var]]),
    levels = c("0", "1"),
    labels = c("No", "Yes")
  )
  train_data[categorical_vars] <- lapply(train_data[categorical_vars], as.factor)
  test_data[categorical_vars] <- lapply(test_data[categorical_vars], as.factor)

  formula <- as.formula(paste(response_var, "~", paste(predictors, collapse = " + ")))

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

  train_probs <- predict(rf_model, newdata = train_data, type = "prob")
  train_data_with_probs <- train_data %>%
    mutate(Predicted_Prob = train_probs$Yes)

  cat("\nTraining set predicted probabilities:\n")
  print(train_data_with_probs[, c("Patient", "Predicted_Prob")])

  average_auc <- mean(rf_model$resample$ROC)
  auc_sd <- sd(rf_model$resample$ROC)
  cat("Training AUC:", round(average_auc, 4), "+/-", round(auc_sd, 4), "\n")

  test_probs <- predict(rf_model, newdata = test_data, type = "prob")
  test_preds <- predict(rf_model, newdata = test_data)
  test_roc <- roc(test_data[[response_var]], test_probs$Yes, levels = c("No", "Yes"))
  test_auc <- pROC::auc(test_roc)
  cat("Test AUC:", round(as.numeric(test_auc), 4), "\n")

  test_cm <- confusionMatrix(test_preds, test_data[[response_var]], positive = "Yes")

  rf_importance <- varImp(rf_model, scale = FALSE)$importance
  rf_importance$Variable <- rownames(rf_importance)
  rf_importance <- rf_importance[, c("Variable", "Overall")]

  output_path <- file.path(results_dir, paste0(output_prefix, "_Summary.csv"))
  write.csv(rf_importance, output_path, row.names = FALSE)
  cat("Variable importance saved to:", output_path, "\n")

  if ("Patient" %in% colnames(train_data) && 42 %in% train_data$Patient) {
    patient_42_probs <- train_data %>%
      filter(Patient == 42) %>%
      mutate(Predicted_Prob = predict(rf_model, newdata = ., type = "prob")$Yes)
    cat("\nPredicted probabilities for Patient 42 (training set):\n")
    print(patient_42_probs[, c("Patient", "Predicted_Prob")])
  }

  pdp_seq <- seq(
    min(train_data[[pdp_col]], na.rm = TRUE),
    max(train_data[[pdp_col]], na.rm = TRUE),
    length.out = 100
  )
  reference_row <- train_data %>%
    dplyr::select(all_of(predictors)) %>%
    summarise(across(
      everything(),
      ~ if (is.numeric(.)) median(., na.rm = TRUE) else as.character(names(sort(table(.), decreasing = TRUE))[1])
    ))

  partial_data <- reference_row[rep(1, length(pdp_seq)), ]
  partial_data[[pdp_col]] <- pdp_seq
  partial_data[categorical_vars] <- lapply(partial_data[categorical_vars], as.factor)

  partial_probs <- predict(rf_model, newdata = partial_data, type = "prob")
  plot_df <- data.frame(x = pdp_seq, Predicted_Prob = partial_probs$Yes)
  plot_df <- plot_df %>% filter(!is.na(Predicted_Prob), Predicted_Prob >= 0, Predicted_Prob <= 1)
  tile_width <- diff(range(plot_df$x)) / max(nrow(plot_df), 1)

  plot <- ggplot(plot_df, aes(x = x, y = Predicted_Prob)) +
    geom_tile(
      aes(fill = x, y = Predicted_Prob / 2, height = Predicted_Prob, width = tile_width),
      alpha = 0.9
    ) +
    geom_line(color = "black", linewidth = 1.4) +
    scale_fill_viridis(name = pdp_col, limits = range(plot_df$x)) +
    scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0.05))) +
    labs(
      title = paste0("Probability of ", response_var, " vs ", pdp_col),
      x = pdp_col,
      y = "Predicted Probability"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "gray80", linewidth = 0.4),
      panel.grid.major.x = element_line(color = "gray90", linewidth = 0.3),
      legend.position = "right"
    )

  print(plot)

  list(
    model = rf_model,
    test_probs = test_probs,
    test_preds = test_preds,
    test_auc = test_auc,
    test_cm = test_cm,
    var_importance = rf_importance
  )
}

model_5mm <- run_rf_model(
  train_data, test_data,
  response_var = response_5mm,
  output_prefix = "RF_LessThan5mm",
  pdp_col = pdp_col,
  results_dir = results_dir
)

model_10mm <- run_rf_model(
  train_data, test_data,
  response_var = response_10mm,
  output_prefix = "RF_LessThan10mm",
  pdp_col = pdp_col,
  results_dir = results_dir
)

invisible(list(model_5mm = model_5mm, model_10mm = model_10mm))
