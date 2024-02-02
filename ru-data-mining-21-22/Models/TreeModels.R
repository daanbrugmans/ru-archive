install.packages("rattle")

library(caret)
library(rattle)
library(MLeval)
library(rpart)
library(randomForest)
library(gbm)


seed(101)

# Decision tree with the whole questionnaire

control <- trainControl(method='repeatedcv', 
                        number=10, 
                        repeats=5)

brfss.df.tree = train(HeartDiseaseorAttack ~ ., 
                  data=train, 
                  method="rpart", 
                  trControl = control,
                  tuneLength = 10)


# Results
fancyRpartPlot(brfss.df.tree$finalModel)

brfss.tree.pred <- predict(brfss.df.tree, newdata=validation, type="raw")
confusionMatrix(brfss.tree.pred, factor(validation$HeartDiseaseorAttack))

# Confusion Matrix of validation set

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No  Yes
# No  7574 2002
# Yes 1628 2766
# 
# Accuracy : 0.7402
# 95% CI : (0.7328, 0.7474)
# No Information Rate : 0.6587
# P-Value [Acc > NIR] : < 2.2e-16
# 
# Kappa : 0.411
# 
# Mcnemar's Test P-Value : 5.981e-10
# 
#            Sensitivity : 0.8231
#            Specificity : 0.5801
#         Pos Pred Value : 0.7909
#         Neg Pred Value : 0.6295
#             Prevalence : 0.6587
#         Detection Rate : 0.5422
#   Detection Prevalence : 0.6855
#      Balanced Accuracy : 0.7016
# 
#       'Positive' Class : No

brfss.tree.pred <- predict(brfss.df.tree, newdata=test, type="raw")
confusionMatrix(brfss.tree.pred, factor(test$HeartDiseaseorAttack))

# Confusion Matrix of test set

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No  Yes
# No  7587 2051
# Yes 1574 2759
# 
# Accuracy : 0.7405
# 95% CI : (0.7332, 0.7478)
# No Information Rate : 0.6557
# P-Value [Acc > NIR] : < 2.2e-16
# 
# Kappa : 0.4115
# 
# Mcnemar's Test P-Value : 2.659e-15
# 
#            Sensitivity : 0.8282
#            Specificity : 0.5736
#         Pos Pred Value : 0.7872
#         Neg Pred Value : 0.6367
#             Prevalence : 0.6557
#         Detection Rate : 0.5431
#   Detection Prevalence : 0.6899
#      Balanced Accuracy : 0.7009
# 
#       'Positive' Class : No
       
brfss.tree.ROC <- data.frame(predict(brfss.df.tree, test, type="prob"))
brfss.tree.ROC$obs <- as.factor(test$HeartDiseaseorAttack)
brfss.tree.ROC$Group <- 'brfss.tree'

rocCurve <- rbind(brfss.tree.ROC)
res <- evalm(rocCurve, title= "ROC Curve decision tree model without tuning")

# Tuned model for improved specificity

# Increase memory limit to overcome RAM limits
# memory.limit(100000)
# memory.limit()

# Do not run! Creates vector size of 13.1GB
# train$HeartDiseaseorAttack <- factor(train$HeartDiseaseorAttack)
# 
# brfss.df.tree.tuned <- randomForest(HeartDiseaseorAttack ~ ., data=train, importance=TRUE,
#                        proximity=TRUE, ntree = 10, replace=TRUE)

tc <- trainControl(
    method = "boot",
    number = 20,
    classProbs = T,
    savePredictions = T
  )

gbmGrid <-  expand.grid(
  interaction.depth = c(1, 2, 5, 8),
  n.trees = seq(0, 60, by = 5),
  shrinkage = 0.1,
  n.minobsinnode = 1
)

brfss.df.tree.tuned = train(HeartDiseaseorAttack ~ ., 
                      data=train, 
                      method="gbm", 
                      trControl = tc,
                      tuneGrid = gbmGrid)

# Results
pretty(brfss.df.tree.tuned$bestTune)

brfss.tree.tuned.pred <- predict(brfss.df.tree.tuned, newdata=validation, type="raw")
confusionMatrix(brfss.tree.tuned.pred, factor(validation$HeartDiseaseorAttack))

# Confusion Matrix of validation set

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No  Yes
# No  7849 1813
# Yes 1353 2955
# 
# Accuracy : 0.7734
# 95% CI : (0.7663, 0.7803)
# No Information Rate : 0.6587
# P-Value [Acc > NIR] : < 2.2e-16
# 
# Kappa : 0.484
# 
# Mcnemar's Test P-Value : 3.42e-16
# 
#            Sensitivity : 0.8530
#            Specificity : 0.6198
#         Pos Pred Value : 0.8124
#         Neg Pred Value : 0.6859
#             Prevalence : 0.6587
#         Detection Rate : 0.5618
#   Detection Prevalence : 0.6916
#      Balanced Accuracy : 0.7364
# 
#       'Positive' Class : No

brfss.tree.tuned.pred <- predict(brfss.df.tree.tuned, newdata=test, type="raw")
confusionMatrix(brfss.tree.tuned.pred, factor(test$HeartDiseaseorAttack))

# Confusion Matrix of test set

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No  Yes
# No  7830 1836
# Yes 1331 2974
# 
# Accuracy : 0.7733
# 95% CI : (0.7663, 0.7802)
# No Information Rate : 0.6557
# P-Value [Acc > NIR] : < 2.2e-16
# 
# Kappa : 0.4851
# 
# Mcnemar's Test P-Value : < 2.2e-16
# 
#            Sensitivity : 0.8547
#            Specificity : 0.6183
#         Pos Pred Value : 0.8101
#         Neg Pred Value : 0.6908
#             Prevalence : 0.6557
#         Detection Rate : 0.5604
#   Detection Prevalence : 0.6919
#      Balanced Accuracy : 0.7365
# 
#       'Positive' Class : No

brfss.tree.tuned.ROC <- data.frame(predict(brfss.df.tree.tuned, test, type="prob"))
brfss.tree.tuned.ROC$obs <- as.factor(test$HeartDiseaseorAttack)
brfss.tree.tuned.ROC$Group <- 'brfss.tree'

rocCurve <- rbind(brfss.tree.tuned.ROC)
res <- evalm(rocCurve, title= "ROC Curve decision tree model with tuning")

print(brfss.df.tree.tuned)

gbmImp <- varImp(brfss.df.tree.tuned, scale = TRUE)
gbmImp

# gbm variable importance
# 
# only 20 most important variables shown (out of 24)
# 
# Overall
# HighBPYes        100.0000
# HighCholYes       39.3165
# DiffWalkYes       33.4618
# StrokeYes         31.4177
# Age80+            23.6511
# GenHlthPoor       20.9974
# GenHlthFair       20.6050
# SexMale           20.1406
# DiabetesDiabetes  16.5863
# Age75 - 79        13.8969
# SmokerYes         10.6232
# GenHlthVery good   9.7804
# Age70 - 74         8.5920
# GenHlthGood        6.5387
# Age65 - 69         6.0225
# Age60 - 64         2.3644
# Age35 - 39         2.3031
# Age40 - 44         0.8068
# Age45 - 49         0.6710
# Age30 - 34         0.6304
