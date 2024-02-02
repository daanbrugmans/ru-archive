seed(101)

# Decision tree with the reduced questionnaire
control <- trainControl(method='repeatedcv', 
                        number=10, 
                        repeats=5)

brfss.df.tree.reduced = train(HeartDiseaseorAttack ~ ., 
                      data=train.reduced, 
                      method="rpart", 
                      trControl = control,
                      tuneLength = 10)

# Results
fancyRpartPlot(brfss.df.tree.reduced$finalModel)

brfss.tree.pred.reduced <- predict(brfss.df.tree.reduced, newdata=validation.reduced, type="raw")
confusionMatrix(brfss.tree.pred.reduced, factor(validation.reduced$HeartDiseaseorAttack))

# Confusion Matrix of validation.reduced set

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No  Yes
# No  7875 2253
# Yes 1327 2515
# 
# Accuracy : 0.7437
# 95% CI : (0.7364, 0.751)
# No Information Rate : 0.6587
# P-Value [Acc > NIR] : < 2.2e-16
# 
# Kappa : 0.4021
# 
# Mcnemar's Test P-Value : < 2.2e-16
# 
#            Sensitivity : 0.8558
#            Specificity : 0.5275
#         Pos Pred Value : 0.7775
#         Neg Pred Value : 0.6546
#             Prevalence : 0.6587
#         Detection Rate : 0.5637
#   Detection Prevalence : 0.7250
#      Balanced Accuracy : 0.6916
# 
#       'Positive' Class : No

brfss.tree.pred.reduced <- predict(brfss.df.tree.reduced, newdata=test.reduced, type="raw")
confusionMatrix(brfss.tree.pred.reduced, factor(test.reduced$HeartDiseaseorAttack))

# Confusion Matrix of test.reduced set

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No  Yes
# No  7853 2275
# Yes 1308 2535
# 
# Accuracy : 0.7435
# 95% CI : (0.7362, 0.7508)
# No Information Rate : 0.6557
# P-Value [Acc > NIR] : < 2.2e-16
# 
# Kappa : 0.4035
# 
# Mcnemar's Test P-Value : < 2.2e-16
# 
#            Sensitivity : 0.8572
#            Specificity : 0.5270
#         Pos Pred Value : 0.7754
#         Neg Pred Value : 0.6596
#             Prevalence : 0.6557
#         Detection Rate : 0.5621
#   Detection Prevalence : 0.7249
#      Balanced Accuracy : 0.6921
# 
#       'Positive' Class : No

brfss.tree.ROC <- data.frame(predict(brfss.df.tree.reduced, validation.reduced, type="prob"))
brfss.tree.ROC$obs <- as.factor(validation.reduced$HeartDiseaseorAttack)
brfss.tree.ROC$Group <- 'brfss.tree'

rocCurve <- rbind(brfss.tree.ROC)
res <- evalm(rocCurve, title= "ROC Curve decision tree model without tuning on reduced variables")

# GBM model on reduced data set
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

brfss.df.tree.tuned.reduced = train(HeartDiseaseorAttack ~ ., 
                            data=train.reduced, 
                            method="gbm", 
                            trControl = tc,
                            tuneGrid = gbmGrid)

# Results
pretty(brfss.df.tree.tuned.reduced$bestTune)

brfss.tree.tuned.reduced.pred <- predict(brfss.df.tree.tuned.reduced, newdata=validation.reduced, type="raw")
confusionMatrix(brfss.tree.tuned.reduced.pred, factor(validation.reduced$HeartDiseaseorAttack))

# Confusion Matrix of validation.reduced set

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No  Yes
# No  7861 1797
# Yes 1341 2971
# 
# Accuracy : 0.7754
# 95% CI : (0.7684, 0.7823)
# No Information Rate : 0.6587
# P-Value [Acc > NIR] : < 2.2e-16
# 
# Kappa : 0.4886
# 
# Mcnemar's Test P-Value : 4.57e-16
# 
#            Sensitivity : 0.8543
#            Specificity : 0.6231
#         Pos Pred Value : 0.8139
#         Neg Pred Value : 0.6890
#             Prevalence : 0.6587
#         Detection Rate : 0.5627
#   Detection Prevalence : 0.6913
#      Balanced Accuracy : 0.7387
# 
#       'Positive' Class : No

brfss.tree.tuned.reduced.pred <- predict(brfss.df.tree.tuned.reduced, newdata=test.reduced, type="raw")
confusionMatrix(brfss.tree.tuned.reduced.pred, factor(test.reduced$HeartDiseaseorAttack))

# Confusion Matrix of test set

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No  Yes
# No  7843 1854
# Yes 1318 2956
# 
# Accuracy : 0.7730
# 95% CI : (0.7659, 0.7799)
# No Information Rate : 0.6557
# P-Value [Acc > NIR] : < 2.2e-16
# 
# Kappa : 0.4835
# 
# Mcnemar's Test P-Value : < 2.2e-16
# 
#            Sensitivity : 0.8561
#            Specificity : 0.6146
#         Pos Pred Value : 0.8088
#         Neg Pred Value : 0.6916
#             Prevalence : 0.6557
#         Detection Rate : 0.5614
#   Detection Prevalence : 0.6941
#      Balanced Accuracy : 0.7353
# 
#       'Positive' Class : No

brfss.tree.tuned.ROC <- data.frame(predict(brfss.df.tree.tuned, validation.reduced, type="prob"))
brfss.tree.tuned.ROC$obs <- as.factor(validation.reduced$HeartDiseaseorAttack)
brfss.tree.tuned.ROC$Group <- 'brfss.tree'

rocCurve <- rbind(brfss.tree.tuned.ROC)
res <- evalm(rocCurve, title= "ROC Curve decision tree model gradient boosting")

gbmImp <- varImp(brfss.df.tree.tuned.reduced, scale = TRUE)
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