#!/usr/bin/env Rscript
#===============================================================================
# Prediction of hand-harvested berry count from within-mask berry detections and
# image-derived plant phenotypes.
#
# For each mask condition (Manual Mask and the seven segmentation models) the
# script fits Ridge regression (glmnet, alpha = 0) and PLSR (pls, SIMPLS) with
# nested leave-one-genotype-out (LOGO) cross-validation: the outer loop holds out
# one genotype, the inner loop tunes lambda (Ridge) or the number of components
# (PLSR) with genotype-grouped folds on the remaining genotypes.
#
# Usage:
#   Rscript harvest_prediction.R --data <harvest table CSV> --out-dir <folder>
#
# Input CSV: one row per (Model, plant) with the columns Model, Image Name,
# Genotype, Observed_Total and every name in predictor_names below. Extra
# columns (for example Immature_gt, Mature_gt) are ignored.
#
# Outputs in --out-dir:
#   logo_predictions.csv, logo_performance.csv, method_comparison.csv,
#   full_predictions.csv, full_fit_performance.csv, final_tuning.csv,
#   performance_table.csv, logo_ridge.png/.pdf, logo_plsr.png/.pdf
#===============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(glmnet)
  library(pls)
  library(ggplot2)
  library(scales)
})


#===============================================================================
# 0. ARGUMENTS
#===============================================================================

parse_arguments <- function(args) {
  
  usage <- "Usage: Rscript harvest_prediction.R --data <CSV> --out-dir <folder>"
  values <- list()
  i <- 1
  
  while (i <= length(args)) {
    
    key <- args[i]
    
    if (key %in% c("-h", "--help")) {
      cat(usage, "\n")
      quit(save = "no", status = 0)
    }
    
    if (!key %in% c("--data", "--out-dir") || i == length(args)) {
      stop(usage, call. = FALSE)
    }
    
    values[[sub("^--", "", key)]] <- args[i + 1]
    i <- i + 2
  }
  
  if (is.null(values[["data"]]) || is.null(values[["out-dir"]])) {
    stop(usage, call. = FALSE)
  }
  
  values
}

arguments <- parse_arguments(commandArgs(trailingOnly = TRUE))
out_dir <- arguments[["out-dir"]]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)


#===============================================================================
# 1. PREPARE PREDICTOR DATA
#===============================================================================

# Within-mask berry count (immature + mature detections) and the 19
# image-derived plant phenotypes.
predictor_names <- c(
  "total_detection",
  "LDI Whole (%)",
  "Std Dev Hue",
  "Mean Hue",
  "Canopy Perimeter (px)",
  "Yellow (%)",
  "Texture ASM",
  "Mean VARI",
  "Canopy Height (px)",
  "Mean Value",
  "Mean Saturation",
  "Orientation",
  "Mean GLI",
  "Canopy Width (px)",
  "Texture Contrast",
  "Texture Energy",
  "Std Dev Saturation",
  "Fractal Dimension",
  "Brown (%)",
  "Hull Area (px\u00b2)"
)

validation_data <- read.csv(
  arguments[["data"]],
  check.names = FALSE,
  stringsAsFactors = FALSE,
  fileEncoding = "UTF-8",
  colClasses = c(
    "Model" = "character",
    "Image Name" = "character",
    "Genotype" = "character"
  )
) %>%
  as_tibble()

missing_columns <- setdiff(
  c("Model", "Image Name", "Genotype", "Observed_Total", predictor_names),
  names(validation_data)
)

if (length(missing_columns) > 0) {
  stop(
    "Input lacks columns: ",
    paste(missing_columns, collapse = ", "),
    call. = FALSE
  )
}

# Desired order for tables and figures.
model_order <- c(
  "Manual Mask",
  "SAM3",
  "SAM2.1",
  "BiRefNet-HR",
  "Center BiRefNet",
  "SegFormer-B2",
  "DeepLabV3+",
  "YOLO26-Seg"
)

prepare_prediction_data <- function(train, test, predictors) {
  
  x_train <- train[, predictors, drop = FALSE]
  x_test <- test[, predictors, drop = FALSE]
  
  # Convert predictors to numeric.
  x_train[] <- lapply(
    x_train,
    function(x) {
      x <- as.numeric(x)
      x[!is.finite(x)] <- NA
      x
    }
  )
  
  x_test[] <- lapply(
    x_test,
    function(x) {
      x <- as.numeric(x)
      x[!is.finite(x)] <- NA
      x
    }
  )
  
  #---------------------------------------------------------------------------
  # Remove predictors containing no usable training observations
  #---------------------------------------------------------------------------
  
  keep <- vapply(
    x_train,
    function(x) any(!is.na(x)),
    logical(1)
  )
  
  x_train <- x_train[, keep, drop = FALSE]
  x_test <- x_test[, keep, drop = FALSE]
  
  #---------------------------------------------------------------------------
  # Impute missing values using training-data medians
  #---------------------------------------------------------------------------
  
  medians <- vapply(
    x_train,
    function(x) median(x, na.rm = TRUE),
    numeric(1)
  )
  
  for (v in names(x_train)) {
    
    x_train[[v]][is.na(x_train[[v]])] <- medians[v]
    x_test[[v]][is.na(x_test[[v]])] <- medians[v]
    
  }
  
  #---------------------------------------------------------------------------
  # Remove zero-variance predictors
  #---------------------------------------------------------------------------
  
  predictor_sd <- vapply(
    x_train,
    sd,
    numeric(1)
  )
  
  keep <- is.finite(predictor_sd) &
    predictor_sd > 0
  
  x_train <- x_train[, keep, drop = FALSE]
  x_test <- x_test[, keep, drop = FALSE]
  
  # Store original predictor names.
  original_names <- names(x_train)
  
  # Convert names to syntactically valid R names for PLSR.
  safe_names <- make.names(
    original_names,
    unique = TRUE
  )
  
  colnames(x_train) <- safe_names
  colnames(x_test) <- safe_names
  
  list(
    x_train = as.matrix(x_train),
    x_test = as.matrix(x_test),
    original_names = original_names,
    safe_names = safe_names
  )
}


#===============================================================================
# 2. NESTED LEAVE-ONE-GENOTYPE-OUT RIDGE REGRESSION
#===============================================================================

ridge_logo <- function(data, predictors) {
  
  data <- data %>%
    filter(
      !is.na(Observed_Total),
      !is.na(Genotype)
    )
  
  genotypes <- unique(
    data$Genotype
  )
  
  if (length(genotypes) < 4) {
    stop(
      "At least four genotypes are recommended for nested LOGO validation."
    )
  }
  
  predictions <- vector(
    "list",
    length(genotypes)
  )
  
  for (i in seq_along(genotypes)) {
    
    #---------------------------------------------------------------------------
    # Outer LOGO split
    #---------------------------------------------------------------------------
    
    test_genotype <- genotypes[i]
    
    train <- data %>%
      filter(
        Genotype != test_genotype
      )
    
    test <- data %>%
      filter(
        Genotype == test_genotype
      )
    
    #---------------------------------------------------------------------------
    # Prepare predictors
    #---------------------------------------------------------------------------
    
    processed <- prepare_prediction_data(
      train = train,
      test = test,
      predictors = predictors
    )
    
    x_train <- processed$x_train
    x_test <- processed$x_test
    
    y_train <- train$Observed_Total
    
    #---------------------------------------------------------------------------
    # If no usable predictors remain, predict training mean
    #---------------------------------------------------------------------------
    
    if (ncol(x_train) == 0) {
      
      predicted <- rep(
        mean(
          y_train,
          na.rm = TRUE
        ),
        nrow(test)
      )
      
      best_lambda <- NA_real_
      
    } else {
      
      #-------------------------------------------------------------------------
      # Inner cross-validation grouped by genotype
      #
      # All observations belonging to the same genotype are placed in the same
      # CV fold.
      #-------------------------------------------------------------------------
      
      inner_genotypes <- unique(
        train$Genotype
      )
      
      foldid <- match(
        train$Genotype,
        inner_genotypes
      )
      
      set.seed(123)
      
      ridge_cv <- cv.glmnet(
        x = x_train,
        y = y_train,
        alpha = 0,
        foldid = foldid,
        standardize = TRUE,
        type.measure = "mse"
      )
      
      # Lambda giving minimum genotype-grouped CV error.
      best_lambda <- ridge_cv$lambda.min
      
      #-------------------------------------------------------------------------
      # Fit ridge model using entire outer training population
      #-------------------------------------------------------------------------
      
      ridge_fit <- glmnet(
        x = x_train,
        y = y_train,
        alpha = 0,
        lambda = best_lambda,
        standardize = TRUE
      )
      
      #-------------------------------------------------------------------------
      # Predict the completely excluded genotype
      #-------------------------------------------------------------------------
      
      predicted <- as.numeric(
        predict(
          ridge_fit,
          newx = x_test,
          s = best_lambda
        )
      )
    }
    
    predictions[[i]] <- tibble(
      Method = "Ridge",
      Model = unique(
        as.character(data$Model)
      ),
      `Image Name` = test$`Image Name`,
      Genotype = test$Genotype,
      Observed_Total = test$Observed_Total,
      Predicted_Total = predicted,
      Lambda = best_lambda,
      Ncomp = NA_integer_
    )
  }
  
  bind_rows(predictions)
}


#===============================================================================
# 3. NESTED LEAVE-ONE-GENOTYPE-OUT PLSR
#===============================================================================

plsr_logo <- function(data, predictors) {
  
  data <- data %>%
    filter(
      !is.na(Observed_Total),
      !is.na(Genotype)
    )
  
  genotypes <- unique(
    data$Genotype
  )
  
  if (length(genotypes) < 4) {
    stop(
      "At least four genotypes are recommended for nested LOGO validation."
    )
  }
  
  predictions <- vector(
    "list",
    length(genotypes)
  )
  
  for (i in seq_along(genotypes)) {
    
    #---------------------------------------------------------------------------
    # Outer LOGO split
    #---------------------------------------------------------------------------
    
    test_genotype <- genotypes[i]
    
    train <- data %>%
      filter(
        Genotype != test_genotype
      )
    
    test <- data %>%
      filter(
        Genotype == test_genotype
      )
    
    #---------------------------------------------------------------------------
    # Prepare predictors
    #---------------------------------------------------------------------------
    
    processed <- prepare_prediction_data(
      train = train,
      test = test,
      predictors = predictors
    )
    
    x_train <- processed$x_train
    x_test <- processed$x_test
    
    y_train <- train$Observed_Total
    
    #---------------------------------------------------------------------------
    # If no predictors remain, predict training mean
    #---------------------------------------------------------------------------
    
    if (ncol(x_train) == 0) {
      
      predicted <- rep(
        mean(
          y_train,
          na.rm = TRUE
        ),
        nrow(test)
      )
      
      best_ncomp <- NA_integer_
      
    } else {
      
      #-------------------------------------------------------------------------
      # Create inner leave-one-genotype-out folds
      #-------------------------------------------------------------------------
      
      inner_segments <- unname(
        split(
          seq_len(nrow(train)),
          train$Genotype
        )
      )
      
      #-------------------------------------------------------------------------
      # Maximum number of PLS components
      #
      # Restrict number of components according to:
      # - number of predictors;
      # - smallest inner training population;
      # - maximum of 15 components.
      #-------------------------------------------------------------------------
      
      largest_validation_fold <- max(
        lengths(inner_segments)
      )
      
      smallest_inner_training <- nrow(train) -
        largest_validation_fold
      
      max_components <- min(
        15,
        ncol(x_train),
        smallest_inner_training - 1
      )
      
      if (max_components < 1) {
        
        predicted <- rep(
          mean(
            y_train,
            na.rm = TRUE
          ),
          nrow(test)
        )
        
        best_ncomp <- NA_integer_
        
      } else {
        
        #-----------------------------------------------------------------------
        # Training dataframe
        #-----------------------------------------------------------------------
        
        pls_train <- data.frame(
          Observed_Total = y_train,
          x_train,
          check.names = FALSE
        )
        
        #-----------------------------------------------------------------------
        # Tune number of PLS components using genotype-grouped CV
        #-----------------------------------------------------------------------
        
        set.seed(123)
        
        pls_cv <- plsr(
          Observed_Total ~ .,
          data = pls_train,
          ncomp = max_components,
          scale = TRUE,
          validation = "CV",
          segments = inner_segments,
          method = "simpls"
        )
        
        #-----------------------------------------------------------------------
        # Extract CV RMSE for each number of components
        #-----------------------------------------------------------------------
        
        pls_rmsep <- RMSEP(
          pls_cv,
          estimate = "CV"
        )
        
        # First value corresponds to the intercept-only model.
        rmsep_values <- drop(
          pls_rmsep$val[
            1,
            1,
            -1
          ]
        )
        
        # Number of components giving minimum CV RMSE.
        best_ncomp <- which.min(
          rmsep_values
        )
        
        #-----------------------------------------------------------------------
        # Fit final PLSR using complete outer training population
        #-----------------------------------------------------------------------
        
        pls_final <- plsr(
          Observed_Total ~ .,
          data = pls_train,
          ncomp = best_ncomp,
          scale = TRUE,
          validation = "none",
          method = "simpls"
        )
        
        pls_test <- data.frame(
          x_test,
          check.names = FALSE
        )
        
        #-----------------------------------------------------------------------
        # Predict excluded genotype
        #-----------------------------------------------------------------------
        
        predicted <- as.numeric(
          predict(
            pls_final,
            newdata = pls_test,
            ncomp = best_ncomp
          )
        )
      }
    }
    
    predictions[[i]] <- tibble(
      Method = "PLSR",
      Model = unique(
        as.character(data$Model)
      ),
      `Image Name` = test$`Image Name`,
      Genotype = test$Genotype,
      Observed_Total = test$Observed_Total,
      Predicted_Total = predicted,
      Lambda = NA_real_,
      Ncomp = best_ncomp
    )
  }
  
  bind_rows(predictions)
}


#===============================================================================
# 4. RUN LOGO VALIDATION FOR ALL SEGMENTATION MODELS
#===============================================================================

logo_predictions <- map_dfr(
  model_order,
  function(current_model) {
    
    model_data <- validation_data %>%
      filter(
        as.character(Model) == current_model
      )
    
    ridge_result <- ridge_logo(
      data = model_data,
      predictors = predictor_names
    )
    
    plsr_result <- plsr_logo(
      data = model_data,
      predictors = predictor_names
    )
    
    bind_rows(
      ridge_result,
      plsr_result
    )
  }
)

logo_predictions <- logo_predictions %>%
  mutate(
    Model = factor(
      Model,
      levels = model_order
    ),
    Method = factor(
      Method,
      levels = c(
        "Ridge",
        "PLSR"
      )
    )
  )


#===============================================================================
# 5. FUNCTION TO CALCULATE PREDICTION PERFORMANCE
#===============================================================================

calculate_performance <- function(predictions) {
  
  predictions %>%
    group_by(
      Method,
      Model
    ) %>%
    summarise(
      
      N = sum(
        complete.cases(
          Observed_Total,
          Predicted_Total
        )
      ),
      
      # Pearson correlation.
      R = cor(
        Observed_Total,
        Predicted_Total,
        use = "complete.obs"
      ),
      
      # Squared correlation.
      R2_cor = R^2,
      
      # Predictive R2.
      # Values can be negative when predictions are worse than predicting
      # the overall observed mean.
      R2_predictive =
        1 -
        sum(
          (
            Observed_Total -
              Predicted_Total
          )^2,
          na.rm = TRUE
        ) /
        sum(
          (
            Observed_Total -
              mean(
                Observed_Total,
                na.rm = TRUE
              )
          )^2,
          na.rm = TRUE
        ),
      
      # Root mean squared error.
      RMSE = sqrt(
        mean(
          (
            Observed_Total -
              Predicted_Total
          )^2,
          na.rm = TRUE
        )
      ),
      
      # Mean absolute error.
      MAE = mean(
        abs(
          Observed_Total -
            Predicted_Total
        ),
        na.rm = TRUE
      ),
      
      .groups = "drop"
    ) %>%
    
    group_by(Method) %>%
    arrange(
      RMSE,
      .by_group = TRUE
    ) %>%
    mutate(
      RMSE_rank = row_number()
    ) %>%
    ungroup()
}


#===============================================================================
# 6. LOGO PREDICTIVE PERFORMANCE
#===============================================================================

logo_performance <- calculate_performance(
  logo_predictions
)



#===============================================================================
# 7. DIRECT RIDGE VS PLSR COMPARISON
#===============================================================================

method_comparison <- logo_performance %>%
  dplyr::select(
    Method,
    Model,
    R,
    R2_cor,
    R2_predictive,
    RMSE,
    MAE
  ) %>%
  pivot_wider(
    names_from = Method,
    values_from = c(
      R,
      R2_cor,
      R2_predictive,
      RMSE,
      MAE
    )
  ) %>%
  mutate(
    
    # Positive value means Ridge has higher RMSE.
    RMSE_difference =
      RMSE_Ridge -
      RMSE_PLSR,
    
    Best_method = case_when(
      RMSE_Ridge < RMSE_PLSR ~ "Ridge",
      RMSE_PLSR < RMSE_Ridge ~ "PLSR",
      TRUE ~ "Equal"
    )
  )



#===============================================================================
# 8. FIT FINAL RIDGE AND PLSR MODELS USING THE COMPLETE DATASET
#===============================================================================

fit_full_models <- function(
    data,
    predictors,
    model_name
) {
  
  data <- data %>%
    filter(
      !is.na(Observed_Total),
      !is.na(Genotype)
    )
  
  #---------------------------------------------------------------------------
  # Prepare complete dataset
  #---------------------------------------------------------------------------
  
  processed <- prepare_prediction_data(
    train = data,
    test = data,
    predictors = predictors
  )
  
  x <- processed$x_train
  y <- data$Observed_Total
  
  #---------------------------------------------------------------------------
  # RIDGE: genotype-grouped CV for lambda
  #---------------------------------------------------------------------------
  
  ridge_foldid <- match(
    data$Genotype,
    unique(data$Genotype)
  )
  
  set.seed(123)
  
  ridge_cv <- cv.glmnet(
    x = x,
    y = y,
    alpha = 0,
    foldid = ridge_foldid,
    standardize = TRUE,
    type.measure = "mse"
  )
  
  best_lambda <- ridge_cv$lambda.min
  
  # Fit final Ridge model using all observations.
  ridge_fit <- glmnet(
    x = x,
    y = y,
    alpha = 0,
    lambda = best_lambda,
    standardize = TRUE
  )
  
  # In-sample fitted values.
  ridge_fitted <- as.numeric(
    predict(
      ridge_fit,
      newx = x,
      s = best_lambda
    )
  )
  
  #---------------------------------------------------------------------------
  # PLSR: genotype-grouped CV for number of components
  #---------------------------------------------------------------------------
  
  pls_segments <- unname(
    split(
      seq_len(nrow(data)),
      data$Genotype
    )
  )
  
  largest_validation_fold <- max(
    lengths(pls_segments)
  )
  
  smallest_training_population <- nrow(data) -
    largest_validation_fold
  
  max_components <- min(
    15,
    ncol(x),
    smallest_training_population - 1
  )
  
  pls_data <- data.frame(
    Observed_Total = y,
    x,
    check.names = FALSE
  )
  
  set.seed(123)
  
  pls_cv <- plsr(
    Observed_Total ~ .,
    data = pls_data,
    ncomp = max_components,
    scale = TRUE,
    validation = "CV",
    segments = pls_segments,
    method = "simpls"
  )
  
  pls_rmsep <- RMSEP(
    pls_cv,
    estimate = "CV"
  )
  
  rmsep_values <- drop(
    pls_rmsep$val[
      1,
      1,
      -1
    ]
  )
  
  best_ncomp <- which.min(
    rmsep_values
  )
  
  # Fit final PLSR using all observations.
  pls_fit <- plsr(
    Observed_Total ~ .,
    data = pls_data,
    ncomp = best_ncomp,
    scale = TRUE,
    validation = "none",
    method = "simpls"
  )
  
  # In-sample fitted values.
  pls_fitted <- as.numeric(
    predict(
      pls_fit,
      newdata = pls_data,
      ncomp = best_ncomp
    )
  )
  
  #---------------------------------------------------------------------------
  # Store tuning parameters
  #---------------------------------------------------------------------------
  
  tuning <- tibble(
    Model = model_name,
    Lambda_min = best_lambda,
    Lambda_1se = ridge_cv$lambda.1se,
    PLS_components = best_ncomp,
    Predictors_used = ncol(x)
  )
  
  #---------------------------------------------------------------------------
  # Store full-data fitted values
  #---------------------------------------------------------------------------
  
  fitted_values <- bind_rows(
    
    tibble(
      Method = "Ridge",
      Model = model_name,
      `Image Name` = data$`Image Name`,
      Genotype = data$Genotype,
      Observed_Total = data$Observed_Total,
      Predicted_Total = ridge_fitted
    ),
    
    tibble(
      Method = "PLSR",
      Model = model_name,
      `Image Name` = data$`Image Name`,
      Genotype = data$Genotype,
      Observed_Total = data$Observed_Total,
      Predicted_Total = pls_fitted
    )
  )
  
  #---------------------------------------------------------------------------
  # Store Ridge coefficients
  #---------------------------------------------------------------------------
  
  ridge_coef <- as.matrix(
    coef(
      ridge_fit,
      s = best_lambda
    )
  )
  
  ridge_coefficients <- tibble(
    Predictor = rownames(ridge_coef),
    Coefficient = as.numeric(
      ridge_coef[, 1]
    )
  )
  
  # Return fitted models and supporting objects.
  list(
    ridge_fit = ridge_fit,
    ridge_cv = ridge_cv,
    pls_fit = pls_fit,
    pls_cv = pls_cv,
    tuning = tuning,
    fitted_values = fitted_values,
    ridge_coefficients = ridge_coefficients,
    predictor_names = processed$original_names
  )
}


#===============================================================================
# 9. FIT COMPLETE DATA FOR MANUAL MASK AND EACH SEGMENTATION MODEL
#===============================================================================

full_model_fits <- setNames(
  map(
    model_order,
    function(current_model) {
      
      model_data <- validation_data %>%
        filter(
          as.character(Model) == current_model
        )
      
      fit_full_models(
        data = model_data,
        predictors = predictor_names,
        model_name = current_model
      )
    }
  ),
  model_order
)


#===============================================================================
# 10. EXTRACT FINAL HYPERPARAMETERS
#===============================================================================

final_tuning <- map_dfr(
  full_model_fits,
  "tuning"
) %>%
  mutate(
    Model = factor(
      Model,
      levels = model_order
    )
  )



#===============================================================================
# 11. EXTRACT FULL-DATA FITTED VALUES
#===============================================================================

full_predictions <- map_dfr(
  full_model_fits,
  "fitted_values"
) %>%
  mutate(
    Model = factor(
      Model,
      levels = model_order
    ),
    Method = factor(
      Method,
      levels = c(
        "Ridge",
        "PLSR"
      )
    )
  )


#===============================================================================
# 12. FULL-DATA APPARENT PERFORMANCE
#===============================================================================

# These statistics describe model fit to the data used to construct the model.
# They should NOT be presented as independent predictive accuracy.

full_fit_performance <- calculate_performance(
  full_predictions
)



#===============================================================================
# 13. FINAL PUBLICATION PERFORMANCE TABLE
#===============================================================================

# LOGO validation is the primary performance assessment.

performance_table <- logo_performance %>%
  left_join(
    final_tuning,
    by = "Model"
  ) %>%
  mutate(
    R = round(
      R,
      3
    ),
    R2_cor = round(
      R2_cor,
      3
    ),
    R2_predictive = round(
      R2_predictive,
      3
    ),
    RMSE = round(
      RMSE,
      1
    ),
    MAE = round(
      MAE,
      1
    ),
    Lambda_min = signif(
      Lambda_min,
      3
    ),
    Lambda_1se = signif(
      Lambda_1se,
      3
    )
  ) %>%
  dplyr::select(
    Method,
    Model,
    N,
    R,
    R2_cor,
    R2_predictive,
    RMSE,
    MAE,
    RMSE_rank,
    Lambda_min,
    Lambda_1se,
    PLS_components,
    Predictors_used
  )



#===============================================================================
# 14. PREPARE LOGO STATISTICS FOR PLOTS
#===============================================================================

plot_statistics <- logo_performance %>%
  mutate(
    label = sprintf(
      "R = %.2f\nR² = %.2f\nRMSE = %.0f",
      R,
      R2_predictive,
      RMSE
    )
  )

# Use common limits for observed and predicted values.
plot_limits <- range(
  c(
    logo_predictions$Observed_Total,
    logo_predictions$Predicted_Total
  ),
  na.rm = TRUE
)


#===============================================================================
# 15. FUNCTION FOR MANUSCRIPT-QUALITY LOGO PLOTS
#===============================================================================

plot_logo_predictions <- function(
    prediction_data,
    statistics_data,
    method_name
) {
  
  p <- ggplot(
    prediction_data %>%
      filter(
        Method == method_name
      ),
    aes(
      x = Observed_Total,
      y = Predicted_Total
    )
  ) +
    
    # Perfect prediction line.
    geom_abline(
      slope = 1,
      intercept = 0,
      linetype = "dashed",
      linewidth = 0.6
    ) +
    
    # Individual LOGO predictions.
    geom_point(
      size = 2,
      alpha = 0.7
    ) +
    
    # Linear relationship between observed and predicted values.
    geom_smooth(
      method = "lm",
      formula = y ~ x,
      se = FALSE,
      linewidth = 0.7
    ) +
    
    # Performance statistics.
    geom_text(
      data = statistics_data %>%
        filter(
          Method == method_name
        ),
      aes(
        x = -Inf,
        y = Inf,
        label = label
      ),
      inherit.aes = FALSE,
      hjust = -0.08,
      vjust = 1.15,
      size = 3.5
    ) +
    
    facet_wrap(
      ~Model,
      ncol = 4
    ) +
    
    scale_x_continuous(
      limits = plot_limits,
      labels = comma,
      expand = expansion(
        mult = 0.08
      )
    ) +
    
    scale_y_continuous(
      limits = plot_limits,
      labels = comma,
      expand = expansion(
        mult = 0.08
      )
    ) +
    
    coord_equal() +
    
    labs(
      x = "Hand-harvested total berry count",
      y = "Predicted total berry count"
    )
  
  p <- p +
    theme_minimal(
      base_size = 13
    ) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(
        linewidth = 0.3
      ),
      panel.border = element_rect(
        fill = NA,
        linewidth = 0.5
      ),
      strip.text = element_text(
        face = "bold",
        size = 11
      ),
      plot.title = element_text(
        face = "bold",
        hjust = 0.5
      ),
      axis.title = element_text(
        size = 13
      ),
      axis.text = element_text(
        size = 10
      ),
      legend.position = "none"
    )
  
  p
}


#===============================================================================
# 16. CREATE RIDGE AND PLSR LOGO FIGURES
#===============================================================================

p_ridge_logo <- plot_logo_predictions(
  prediction_data = logo_predictions,
  statistics_data = plot_statistics,
  method_name = "Ridge"
)

p_plsr_logo <- plot_logo_predictions(
  prediction_data = logo_predictions,
  statistics_data = plot_statistics,
  method_name = "PLSR"
)


ggsave(
  file.path(out_dir, "logo_ridge.png"),
  p_ridge_logo,
  width = 14,
  height = 8,
  dpi = 300,
  bg = "white"
)
ggsave(
  file.path(out_dir, "logo_ridge.pdf"),
  p_ridge_logo,
  width = 14,
  height = 8
)
ggsave(
  file.path(out_dir, "logo_plsr.png"),
  p_plsr_logo,
  width = 14,
  height = 8,
  dpi = 300,
  bg = "white"
)
ggsave(
  file.path(out_dir, "logo_plsr.pdf"),
  p_plsr_logo,
  width = 14,
  height = 8
)


#===============================================================================
# 17. WRITE TABLES
#===============================================================================

write_table <- function(table, name) {
  write.csv(
    as.data.frame(table),
    file.path(out_dir, name),
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
}

write_table(logo_predictions, "logo_predictions.csv")
write_table(logo_performance, "logo_performance.csv")
write_table(method_comparison, "method_comparison.csv")
write_table(full_predictions, "full_predictions.csv")
write_table(full_fit_performance, "full_fit_performance.csv")
write_table(final_tuning, "final_tuning.csv")
write_table(performance_table, "performance_table.csv")

cat("Wrote", out_dir, "\n")
