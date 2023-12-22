library('readr')
library('dplyr')
library('e1071')
library('corrplot')
library(ggplot2)
library(caret)
library(visdat)

#import data
data <- read.csv("C:/Users/ravih/Downloads/Loan.csv", header=TRUE)
summary(data)

vis_dat(data)

# Check for missing values in the entire dataset
missing_values <- sum(is.na(data))
# Display the number of missing values
cat("Number of missing values in the dataset:", missing_values, "\n")

# Check for duplicate rows in the entire dataset
duplicate_rows <- data[duplicated(data), ]
print(duplicate_rows)

#removing  loan id because every id is unique
data <- data[, !(colnames(data) == "LoanID")]
str(data)
dim(data)


# Separate numerical and categorical columns
numerical_cols <- data %>%
  select_if(is.numeric)
dim(numerical_cols)
numerical_cols<-numerical_cols[, -which(names(numerical_cols) %in% "Default")]


categorical_cols <- data %>%
  select_if(is.character)
dim(categorical_cols)


# Display the structure of numerical and categorical data
str(numerical_cols)
str(categorical_cols)




######barplots for categorical variables
par(mfrow= c(2,4))

for (col in c(names(categorical_cols))){
  categorical_cols %>% pull(col) %>% table %>% barplot(main= col)
  
} 


# Create dummy variables for categorical columns using model.matrix
dummy_data <- as.data.frame(model.matrix(~ . - 1, data = categorical_cols))
head(dummy_data)




# Display the structure of the updated dataframe with dummy variables
dim(dummy_data)









###checking Nearzerovar

# Check for near-zero variance in numerical columns
near_zero_vars <- nearZeroVar(dummy_data, saveMetrics = TRUE)
# Display the near-zero variance variables
print(near_zero_vars)


# Append dummy variables to the original dataset
loan_data<- cbind(numerical_cols, dummy_data)
##loan_data<-loan_data[, -which(names(loan_data) %in% "Default")]

# Display the structure of the updated dataset
str(loan_data)
dim(loan_data)







#create histogram and boxplot for each numerical column
par(mfrow= c(3,3))
for (col in c(names(numerical_cols))){
  numerical_cols %>% pull(col) %>% hist(main= col)
}
for (col in c(names(numerical_cols))){
  numerical_cols %>% pull(col) %>% boxplot(main= col)
}





###correlation matrix########
# Calculate correlation matrix
library(corrplot)
cor_matrix <- cor(loan_data)
dev.new()
dim(loan_data)

# Create a correlation plot
corrplot(cor_matrix, method = "circle", diag = TRUE, tl.cex = 0.8)

# Find highly correlated variables (correlation greater than 0.80)
highly_correlated <- which(upper.tri(cor_matrix, diag = TRUE) & cor_matrix > 0.80, arr.ind = TRUE)

# Print highly correlated variable pairs and their correlation values
for (i in 1:nrow(highly_correlated)) {
  var1 <- rownames(cor_matrix)[highly_correlated[i, 1]]
  var2 <- colnames(cor_matrix)[highly_correlated[i, 2]]
  correlation_value <- cor_matrix[highly_correlated[i, 1], highly_correlated[i, 2]]
  
  cat(sprintf("Variables %s and %s are highly correlated (correlation = %.2f)\n", var1, var2, correlation_value))
}
names(loan_data)


######splitting data 
splitIndex <- createDataPartition(data$Default, p = 0.8, list = FALSE, times = 1)

# Split data into training and testing sets
training_data <- loan_data[splitIndex, ]
testing_data <- loan_data[-splitIndex, ]

training_Default <- data$Default[splitIndex]
testing_Default <- data$Default[-splitIndex]


# Convert response variable to a factor with two levels
training_Default <- as.factor(training_Default)
testing_Default <- as.factor(testing_Default)


# Check the levels of your factor variable
levels(training_Default)


# Change levels from "0" to "No" and from "1" to "Yes"
levels(training_Default) <- c("No", "Yes")
levels(testing_Default) <- c("No", "Yes")

# Verify that the levels have been changed
levels(training_Default)
levels(testing_Default)




################## Models building
####### Logistic Regression
set.seed(123)
ctrl <- trainControl(method = "cv", number = 5,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

lrFull <- train(x= training_data,
                y = training_Default,
                method = "glm",
                family = "binomial",
                metric = "ROC" ,
                preProc = c("center", "scale"),
                trControl = ctrl)
lrFull

library(pROC)
FullRoc <- roc(lrFull$pred$obs,lrFull$pred$Yes)
plot(FullRoc, legacy.axes = TRUE, col = "blue", main = "ROC Curve")
auc(FullRoc)

lrPred <- predict(lrFull,newdata = testing_data)
confusionMatrix(lrPred,testing_Default)
#pred <- predict(glmnTuned, newdata= test_data, type="raw")
roc(testing_Default, as.numeric(lrPred))





#######LDA#####
####### LDA 
## Using train function, should add pre-processing

ctrl <- trainControl(method = "cv", number = 5,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

set.seed(123)
LDAFull <- train(x = training_data,
                 y = training_Default,
                 method = "lda",
                 metric = "ROC",
                 trControl = ctrl)
LDAFull
summary(LDAFull)
plot(LDAFull)


ldaPred <- predict(LDAFull,newdata = test_data)
confusionMatrix(ldaPred,test_response)


library(pROC)
FullRoc <- roc(LDAFull$pred$obs,LDAFull$pred$Yes)
plot(FullRoc, legacy.axes = TRUE, col = "blue", main = "ROC Curve")
auc(FullRoc)







#####SVM

# Set up the training control with ROC as the summary function
ctrl <- trainControl(method = "cv", number= 5, summaryFunction = twoClassSummary, classProbs = TRUE)

set.seed(100)
svm_model <- train(x = training_data,
                   y = training_Default,
                   method = "svmRadial",
                   metric = "ROC",
                   preProc = c("center", "scale"),
                   tuneLength = 10,
                   trControl = ctrl)


svm_model
plot(svm_model)
ggplot(svm_model)+coord_trans(x='log2')

svmRpred <- predict(svm_model, newdata = test_data)
confusionMatrix(svmRpred,test_response)

svmRaccuracy <- data.frame(obs = test_response , pred = svmRpred)
defaultSummary(svmRaccuracy)





# Make predictions on the test data
svmRpred <- predict(svm_model, newdata = test_data)

# Evaluate the model using confusion matrix and other metrics
confusionMatrix(svmRpred, test_response)

# Create ROC curve
svm_probs <- predict(svm_model, newdata = test_data, type = "prob")[, "Yes"]
FullRoc <- roc(test_response, svm_probs)

# Plot the ROC curve
plot(FullRoc, legacy.axes = TRUE, col = "blue", main = "ROC Curve")

# Print AUC
auc_value <- auc(FullRoc)
cat("AUC:", auc_value, "\n")





######KNN

ctrl<- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

set.seed(123)
knnTune <- train(x = training_data,
                 y = training_Default,
                 method = "knn",
                 metric = "ROC",
                 # Center and scaling will occur for new predictions too
                 preProc = c("center", "scale"),
                 tuneGrid = data.frame(.k = 1:15),
                 trControl = ctrl)
knnTune
plot(knnTune)


knnpred <- predict(knnTune, newdata = test_data)
confusionMatrix(knnpred,test_response)

knnaccuracy <- data.frame(obs = test_response , pred = knnpred)
defaultSummary(knnaccuracy)

library(pROC)
FullRoc <- roc(knnTune$pred$obs,knnTune$pred$Yes)
plot(FullRoc, legacy.axes = TRUE, col = "blue", main = "ROC Curve")
auc(FullRoc)



###### Neural Networks
nnetGrid <- expand.grid(.decay = c(0, 0.01, .1),
                        .size = c(1:10),
                        ## The next option is to use bagging (see the
                        ## next chapter) instead of different random
                        ## seeds.
                        .bag = T)

ctrl <- trainControl(method = "cv", number = 5, classProbs = T, summaryFunction = twoClassSummary)

set.seed(123)
nnetTune <- train(training_data, training_Default,
                  method = "avNNet",
                  metric = "ROC", 
                  tuneGrid = nnetGrid,
                  trControl = ctrl,
                  ## Automatically standardize data prior to modeling
                  ## and prediction
                  preProc = c("center", "scale"),
                  linout = TRUE,
                  trace = FALSE,
                  MaxNWts = 10 * (ncol(training_data) + 1) + 10 + 1,
                  maxit = 500)


nnetTune
plot(nnetTune)





######PLSDA

ctrl <- trainControl(method= "cv", number= 5, summaryFunction = twoClassSummary,
                     classProbs = TRUE)

## caret contains a built-in function called twoClassSummary that calculates the
## area under the ROC curve, the sensitivity, and the specificity.
set.seed(123)
plsFit2 <- train(x = training_data,
                 y = training_Default,
                 method = "pls",
                 tuneGrid = expand.grid(.ncomp = 1:10),
                 preProc = c("center","scale"),
                 metric = "ROC",
                 trControl = ctrl)
plsFit2
plot(plsFit2)




#####Penalized 
ctrl <- trainControl(method = "cv", number= 5, 
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     ##index = list(simulatedTest[,1:4]),
                     savePredictions = TRUE)
set.seed(123)

glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 10))
set.seed(123)
glmnTuned <- train(x=training_data,
                   y = training_Default,
                   method = "glmnet",
                   tuneGrid = glmnGrid,
                   preProc = c("center", "scale"),
                   metric = "ROC",
                   trControl = ctrl)
glmnTuned
plot(glmnTuned)



glmnpred <- predict(glmnTuned,newdata = testing_data)
confusionMatrix(glmnpred,testing_Default)
#pred <- predict(glmnTuned, newdata= test_data, type="raw")
roc(testing_Default, as.numeric(glmnpred))




########### Nearest Shrunken Centroids
ctrl <- trainControl(method= "cv", number= 5, summaryFunction = twoClassSummary,
                     classProbs = TRUE)

## nscGrid <- data.frame(.threshold = 0:4)
nscGrid <- data.frame(.threshold = seq(0,4, by=0.1))

set.seed(123)
nscTuned <- train(x=training_data,
                  y = training_Default,
                  method = "pam",
                  preProc = c("center", "scale"),
                  tuneGrid = nscGrid,
                  metric = "ROC",
                  trControl = ctrl)

nscTuned
plot(nscTuned)



############ RDA 
ctrl <- trainControl(method = "cv", number = 5,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

set.seed(123)
rdaGrid <- expand.grid(.gamma= 1:10, .lambda = c(0, .1, 1, 2))
rdaFit <- train(x=training_data,
                y = training_Default,
                method = "rda",
                metric = "ROC",
                tuneGrid = rdaGrid,
                trControl = ctrl)
rdaFit
plot(rdaFit)
rda_Pred<- predict(rdaFit, newdata=testing_data)
confusionMatrix(data=rda_Pred, reference =testing_labels)






########## Flexible Discriminant Analysis
marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:38)

ctrl<- trainControl(method = "cv", number = 5, summaryFunction =twoClassSummary, classProbs = T )

set.seed(123)
fdaTuned <- train(x=training_data,
                  y = training_Default,
                  method = "fda",
                  metric = "ROC",
                  preProc = c("center", "scale"),
                  # Explicitly declare the candidate models to test
                  tuneGrid = marsGrid,
                  trControl = ctrl)

fdaTuned
plot(fdaTuned)
plot(fdaTuned,main="FDA, degree = 1 and nprune = 6")
fdaPred <- predict(fdaTuned, newdata = simulatedTest[,1:4])
confusionMatrix(data = fdaPred,reference =simulatedTest[,6])




########## Naive Bayes 
install.packages("klaR")
library(klaR)

ctrl<- trainControl(method = "cv", number = 5, summaryFunction = twoClassSummary, classProbs = T) 

set.seed(123)
nbFit <- train( x=training_data,
                y = training_Default,
                method = "nb",
                metric = "ROC",
                preProc = c("center", "scale"),
                ##tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)), ## 21 is the best
                tuneGrid = data.frame(.fL = 2,.usekernel = TRUE,.adjust = TRUE),
                trControl = ctrl)

nbFit
plot(nbFit)




####### Variable importance of best models
imp_predictors <- varImp(lrFull)
imp_predictors
plot(imp_predictors, top = 5)

####### Variable importance of best models
imp_predictors <- varImp(glmnTuned)
imp_predictors
plot(imp_predictors, top = 5)










