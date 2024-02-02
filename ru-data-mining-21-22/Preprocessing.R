rm(list=ls())

library(ggplot2)
library(scales)
library(dplyr)
library(caret)

# Change working directory if needed, Josse.
setwd("C:/Users/Daan/Documents/Projecten/DataMiningResearch")

brfss.df <- read.csv("BRFSS2015.csv", header = T)

summary(brfss.df)


# Change binary data still labeled as numeric as categorical
brfss.df$HeartDiseaseorAttack <- as.factor(brfss.df$HeartDiseaseorAttack)

brfss.df$HighBP <- as.factor(brfss.df$HighBP)

brfss.df$HighChol <- as.factor(brfss.df$HighChol)

brfss.df$CholCheck <- as.factor(brfss.df$CholCheck)

brfss.df$Smoker <- as.factor(brfss.df$Smoker)

brfss.df$Stroke <- as.factor(brfss.df$Stroke)

brfss.df$PhysActivity <- as.factor(brfss.df$PhysActivity)

brfss.df$Fruits <- as.factor(brfss.df$Fruits)

brfss.df$Veggies <- as.factor(brfss.df$Veggies)

brfss.df$HvyAlcoholConsump <- as.factor(brfss.df$HvyAlcoholConsump)

brfss.df$AnyHealthcare <- as.factor(brfss.df$AnyHealthcare)

brfss.df$NoDocbcCost <- as.factor(brfss.df$NoDocbcCost)

brfss.df$DiffWalk <- as.factor(brfss.df$DiffWalk)

brfss.df$Sex <- as.factor(brfss.df$Sex)

summary(brfss.df)


# Change remaining categorical data still labeled as numeric to categorical
brfss.df$Diabetes <- as.factor(brfss.df$Diabetes)

brfss.df$GenHlth <- as.factor(brfss.df$GenHlth)

brfss.df$MentHlth <- as.factor(brfss.df$MentHlth)

brfss.df$PhysHlth <- as.factor(brfss.df$PhysHlth)

brfss.df$Age <- as.factor(brfss.df$Age)

brfss.df$Education <- as.factor(brfss.df$Education)

brfss.df$Income <- as.factor(brfss.df$Income)

summary(brfss.df)


# Converting BMI from continuous value to categorical value.
  # This is performed by taking every BMI value and placing it in one of four categories:
    # 1. Underweight (BMI < 18.5)
    # 2. Healthy weight (18.5 <= BMI < 25)
    # 3. Overweight (25 <= BMI < 30)
    # 4. Class 1 Obese (30 <= BMI < 35)
    # 5. Class 2 Obese (35 <= BMI < 40)
    # 6. Class 3 Obese (BMI >= 40)
  # Categories taken from CDC (https://www.cdc.gov/obesity/adult/defining.html)
BMI.to.categorical <- function(BMI.value){
  if (BMI.value >= 40) {
    BMI.value <- "Class 3 Obese"
  } else if (BMI.value >= 35) {
    BMI.value <- "Class 2 Obese"
  } else if (BMI.value >= 30) {
    BMI.value <- "Class 1 Obese"
  } else if (BMI.value >= 25) {
    BMI.value <- "Overweight"
  } else if (BMI.value >= 18.5) {
    BMI.value <- "Healthy weight"
  } else {
    BMI.value <- "Underweight"
  }
}

brfss.df$BMI <- sapply(brfss.df$BMI, BMI.to.categorical)
summary(brfss.df)


# Give the following features more readable names: Diabetes, GenHlth, Sex, Age, Education, Income
brfss.df <- brfss.df %>%
  mutate(Diabetes = recode(Diabetes, "0" = "No diabetes", "1" = "Pre-diabetes", "2" = "Diabetes")) %>%
  mutate(GenHlth = recode(GenHlth, "1"="Excellent", "2"="Very good", "3"="Good", "4"="Fair", "5"="Poor")) %>%
  mutate(Sex = recode(Sex, "0"="Female", "1"="Male")) %>%
  mutate(Age = recode(Age, "1"="18 - 24", "2"="25 - 29", "3"="30 - 34", "4"="35 - 39", "5"="40 - 44", "6"="45 - 49", "7"="50 - 54", "8"="55 - 59", "9"="60 - 64", "10"="65 - 69", "11"="70 - 74", "12"="75 - 79", "13"="80+")) %>%
  mutate(Education = recode(Education, "1"="Never attended school or only kindergarten", "2"="Elementary school", "3"="Some high school", "4"="High school graduate", "5"="Some college or technical school", "6"="College graduate")) %>%
  mutate(Income = recode(Income, "1"="Income < $10,000", "2"="$10,000 <= Income < $15,000", "3"="$15,000 <= Income < $20,000", "4"="$20,000 <= Income < $25,000", "5"="$25,000 <= Income < $35,000", "6"="$35,000 <= Income < $50,000", "7"="$50,000 <= Income < $75,000", "8"="Income >= $75,000"))
  
summary(brfss.df)


# Change value names for binary values from "0/1" to "Yes/No"
brfss.df <- brfss.df %>%
  mutate(HeartDiseaseorAttack = recode(HeartDiseaseorAttack, "0"="No", "1"="Yes")) %>%
  mutate(HighBP = recode(HighBP, "0"="No", "1"="Yes")) %>%
  mutate(HighChol = recode(HighChol, "0"="No", "1"="Yes")) %>%
  mutate(CholCheck = recode(CholCheck, "0"="No", "1"="Yes")) %>%
  mutate(Smoker = recode(Smoker, "0"="No", "1"="Yes")) %>%
  mutate(Stroke = recode(Stroke, "0"="No", "1"="Yes")) %>%
  mutate(PhysActivity = recode(PhysActivity, "0"="No", "1"="Yes")) %>%
  mutate(Fruits = recode(Fruits, "0"="No", "1"="Yes")) %>%
  mutate(Veggies = recode(Veggies, "0"="No", "1"="Yes")) %>%
  mutate(HvyAlcoholConsump = recode(HvyAlcoholConsump, "0"="No", "1"="Yes")) %>%
  mutate(AnyHealthcare = recode(AnyHealthcare, "0"="No", "1"="Yes")) %>%
  mutate(NoDocbcCost = recode(NoDocbcCost, "0"="No", "1"="Yes")) %>%
  mutate(DiffWalk = recode(DiffWalk, "0"="No", "1"="Yes"))

summary(brfss.df)


# Write preprocessed data frame to CSV file
write.csv(x=brfss.df, file="BRFSS2015Preprocessed.csv")

