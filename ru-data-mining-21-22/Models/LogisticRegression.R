library(caret)
library(MLeval)

set.seed(101)


# Linear regression model, all features, with tenfold cross-validation
control <- trainControl(method="repeatedcv",
                        number=10,
                        repeats=5)

brfss.df.lm <- train(HeartDiseaseorAttack ~ .,
                     data=train,
                     method="glm",
                     trControl=control)

brfss.df.lm$finalModel

brfss.df.lm.pred <- predict(brfss.df.lm, 
                            newdata=validation, 
                            type="raw") # Set to "prob" for logistic regression probabilities

confusionMatrix(brfss.df.lm.pred,
                factor(validation$HeartDiseaseorAttack))

glmImp <- varImp(brfss.df.lm, scale=T)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction    No   Yes
# No  11789  2879
# Yes  1942  4345
# 
# Accuracy : 0.7699          
# 95% CI : (0.7642, 0.7756)
# No Information Rate : 0.6553          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.4746          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
#                                           
#             Sensitivity : 0.8586          
#             Specificity : 0.6015          
#          Pos Pred Value : 0.8037          
#          Neg Pred Value : 0.6911          
#              Prevalence : 0.6553          
#          Detection Rate : 0.5626          
#    Detection Prevalence : 0.7000          
#       Balanced Accuracy : 0.7300          
#                                           
#        'Positive' Class : No         

brfss.df.lm.ROC <- data.frame(predict(brfss.df.lm, newdata=validation, type="prob"))
brfss.df.lm.ROC$obs <- as.factor(validation$HeartDiseaseorAttack)
brfss.df.lm.ROC$Group <- "brfss.df.lm"

ROC <- rbind(brfss.df.lm.ROC)
ROC.plot <- evalm(ROC, title="ROC curve of a logistic regression model\nwith access to all features trained using 10-fold cross-validation")


# Linear regression model, all features, with boosting
control <- trainControl(
  method = "boot",
  number = 40,
  classProbs = T,
  savePredictions = T
)

brfss.df.lm <- train(HeartDiseaseorAttack ~ .,
                     data=train,
                     method="glm",
                     trControl=control)

brfss.df.lm$finalModel

brfss.df.lm.pred <- predict(brfss.df.lm, 
                            newdata=validation, 
                            type="raw") # Set to "prob" for logistic regression probabilities

confusionMatrix(brfss.df.lm.pred,
                factor(validation$HeartDiseaseorAttack))

glmImp <- varImp(brfss.df.lm, scale=T)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction    No   Yes
# No  11789  2879
# Yes  1942  4345
# 
# Accuracy : 0.7699          
# 95% CI : (0.7642, 0.7756)
# No Information Rate : 0.6553          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.4746          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
#                                           
#             Sensitivity : 0.8586          
#             Specificity : 0.6015          
#          Pos Pred Value : 0.8037          
#          Neg Pred Value : 0.6911          
#              Prevalence : 0.6553          
#          Detection Rate : 0.5626          
#    Detection Prevalence : 0.7000          
#       Balanced Accuracy : 0.7300          
#                                           
#        'Positive' Class : No  

brfss.df.lm.ROC <- data.frame(predict(brfss.df.lm, newdata=validation, type="prob"))
brfss.df.lm.ROC$obs <- as.factor(validation$HeartDiseaseorAttack)
brfss.df.lm.ROC$Group <- "brfss.df.lm"

ROC <- rbind(brfss.df.lm.ROC)
ROC.plot <- evalm(ROC, title="ROC curve of a logistic regression model\nwith access to all features trained using boosting")


# Linear regression model, selected features, with tenfold cross-validation
control <- trainControl(method="repeatedcv",
                        number=10,
                        repeats=5)

brfss.df.lm <- train(HeartDiseaseorAttack ~ .,
                     data=train.reduced,
                     method="glm",
                     trControl=control)

brfss.df.lm$finalModel

brfss.df.lm.pred <- predict(brfss.df.lm, 
                            newdata=validation.reduced, 
                            type="raw") # Set to "prob" for logistic regression probabilities

confusionMatrix(brfss.df.lm.pred,
                factor(validation.reduced$HeartDiseaseorAttack))

glmImp <- varImp(brfss.df.lm, scale=T)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction    No   Yes
# No  11830  2907
# Yes  1901  4317
# 
# Accuracy : 0.7706          
# 95% CI : (0.7648, 0.7762)
# No Information Rate : 0.6553          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.4748          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
#                                           
#             Sensitivity : 0.8616          
#             Specificity : 0.5976          
#          Pos Pred Value : 0.8027          
#          Neg Pred Value : 0.6943          
#              Prevalence : 0.6553          
#          Detection Rate : 0.5645          
#    Detection Prevalence : 0.7033          
#       Balanced Accuracy : 0.7296          
#                                           
#        'Positive' Class : No 

brfss.df.lm.ROC <- data.frame(predict(brfss.df.lm, newdata=validation.reduced, type="prob"))
brfss.df.lm.ROC$obs <- as.factor(validation.reduced$HeartDiseaseorAttack)
brfss.df.lm.ROC$Group <- "brfss.df.lm"

ROC <- rbind(brfss.df.lm.ROC)
ROC.plot <- evalm(ROC, title="ROC curve of a logistic regression model\nwith access to selected features trained using 10-fold cross-validation")

