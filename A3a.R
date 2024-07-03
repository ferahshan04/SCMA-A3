#Install packages
install.packages("caret")
install.packages("rpart.plot")
install.packages("glmnet")

# Load necessary libraries
library(caret)
library(pROC)
library(rpart)
library(rpart.plot)
library(glmnet)  # For regularization

# Load your dataset
df <- read.csv("C:\\Users\\Ferah Shan\\Downloads\\kidney_disease.csv")

# Display the first few rows of the dataset
print(head(df))

# Summary statistics of the dataset
print(summary(df))

# Check for missing values
cat("Total missing values: ", sum(is.na(df)), "\n")

# Custom function to calculate mode
mode_function <- function(x) {
  uniq_x <- unique(x)
  uniq_x[which.max(tabulate(match(x, uniq_x)))]
}

# Function to impute missing values
impute_missing_values <- function(df) {
  for (col in names(df)) {
    if (is.numeric(df[[col]])) {
      df[[col]][is.na(df[[col]])] <- median(df[[col]], na.rm = TRUE)
    } else {
      df[[col]][is.na(df[[col]])] <- mode_function(df[[col]][!is.na(df[[col]])])
    }
  }
  return(df)
}

# Impute missing values
df <- impute_missing_values(df)

# Verify there are no more missing values
cat("Total missing values after imputation: ", sum(is.na(df)), "\n")

# Ensure the target variable is a factor with exactly two levels
df$class <- as.factor(df$class)

# Convert target variable to numeric (1 for "ckd" and 0 for "notckd")
df$class <- ifelse(df$class == "ckd", 1, 0)

# Feature scaling
preProc <- preProcess(df[, -which(names(df) == "class")], method = c("center", "scale"))
scaled_data <- predict(preProc, df[, -which(names(df) == "class")])
df_scaled <- cbind(scaled_data, class = df$class)

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(df_scaled$class, p = 0.7, list = FALSE)
trainData <- df_scaled[trainIndex,]
testData <- df_scaled[-trainIndex,]

# Check the distribution of the target variable in training and testing sets
cat("Training set distribution:\n")
print(table(trainData$class))
cat("Testing set distribution:\n")
print(table(testData$class))

# Prepare data for glmnet (regularized logistic regression)
x_train <- as.matrix(trainData[, -which(names(trainData) == "class")])
x_test <- as.matrix(testData[, -which(names(testData) == "class")])

y_train <- as.numeric(trainData$class)
y_test <- as.numeric(testData$class)

# Ensure all features are numeric in x_train
non_numeric_columns <- colnames(trainData)[!sapply(trainData, is.numeric)]
cat("Non-numeric columns in trainData: ", non_numeric_columns, "\n")

# If there are non-numeric columns, convert them to numeric
for (col in non_numeric_columns) {
  suppressWarnings({
    trainData[[col]] <- as.numeric(as.character(trainData[[col]]))
    testData[[col]] <- as.numeric(as.character(testData[[col]]))
  })
}

# Re-prepare the matrices after conversion
x_train <- as.matrix(trainData[, -which(names(trainData) == "class")])
x_test <- as.matrix(testData[, -which(names(testData) == "class")])

# Check for NA values in the matrices and vectors
cat("Any NA in x_train after conversion: ", any(is.na(x_train)), "\n")
cat("Any NA in x_test after conversion: ", any(is.na(x_test)), "\n")

# If there are still missing values, use makeX() to impute them
x_train[is.na(x_train)] <- 0
x_test[is.na(x_test)] <- 0

# Logistic Regression Model with Regularization
log_model <- glmnet(x_train, y_train, family = "binomial")

# Cross-validation to select the best lambda
cv_log_model <- cv.glmnet(x_train, y_train, family = "binomial")

# Predict on the test set using the best lambda
log_pred <- predict(cv_log_model, newx = x_test, s = "lambda.min", type = "response")
log_pred <- as.vector(log_pred)  # Ensure log_pred is a numeric vector
log_pred_class <- ifelse(log_pred > 0.5, 1, 0)

# Confusion Matrix for Logistic Regression
log_conf_matrix <- confusionMatrix(as.factor(log_pred_class), as.factor(y_test))
cat("Confusion Matrix for Logistic Regression:\n")
print(log_conf_matrix)

# ROC Curve for Logistic Regression
roc_log <- roc(y_test, log_pred)
plot(roc_log, main = "ROC Curve for Logistic Regression", col = "black")
cat("AUC for Logistic Regression: ", auc(roc_log), "\n")

# Decision Tree Model
tree_model <- rpart(class ~ ., data = trainData, method = "class")

# Plot Decision Tree
rpart.plot(tree_model)

# Predict on the test set using Decision Tree
tree_pred <- predict(tree_model, newdata = testData, type = "class")
tree_pred_prob <- predict(tree_model, newdata = testData, type = "prob")[, 2]

# Confusion Matrix for Decision Tree
tree_conf_matrix <- confusionMatrix(tree_pred, as.factor(y_test))
cat("Confusion Matrix for Decision Tree:\n")
print(tree_conf_matrix)

# ROC Curve for Decision Tree
roc_tree <- roc(y_test, tree_pred_prob)
plot(roc_tree, col = "red", add = TRUE)
legend("bottomright", legend = c("Logistic Regression", "Decision Tree"), col = c("black", "red"), lwd = 2)
cat("AUC for Decision Tree: ", auc(roc_tree), "\n")

# Compare Models
cat("Logistic Regression vs Decision Tree\n")
cat("AUC for Logistic Regression: ", auc(roc_log), "\n")
cat("AUC for Decision Tree: ", auc(roc_tree), "\n")