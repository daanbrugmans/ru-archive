rm(list=ls())
setwd("C:/Users/Daan/Documents/Projecten/ru-bayesian-networks-and-causal-inference-23-24") # Repository root

library(tidyverse)

path_to_unaltered_dataset = paste(getwd(), "/data/bank-full.csv", sep="")
unaltered_data <- read.csv(path_to_unaltered_dataset, sep=";", stringsAsFactors=T)

  # Selecting and renaming variables
banking_dataset <- unaltered_data %>%
  rename(Age = age, AnnualBalance = balance, JobCategory = job,
         MaritalStatus = marital, EducationLevel = education,
         HasDefault = default, HasHousingLoan = housing, HasPersonalLoan = loan,
         CallDuration = duration, CurrentCampaignCalls = campaign,
         PreviousCampaignsCalls = previous, PreviousCampaignOutcome = poutcome,
         HasSubscribedToDeposit = y) %>%
  select(Age, AnnualBalance, CallDuration, CurrentCampaignCalls, EducationLevel,
         HasDefault, HasHousingLoan, HasPersonalLoan, JobCategory,
         MaritalStatus, PreviousCampaignsCalls, PreviousCampaignOutcome,
         HasSubscribedToDeposit)

  # Removing missing values from set variables
banking_dataset <- banking_dataset[!(banking_dataset$EducationLevel == "unknown" | banking_dataset$JobCategory=="unknown"),]

  # Mark levels of ordinal variables
banking_dataset$EducationLevel <- factor(banking_dataset$EducationLevel, levels=c("primary", "secondary", "tertiary"))

  # Replace character values of binary variables with numeric values
binary_factor_to_numeric <- function(df_feature) {
  df_feature_numeric <- as.numeric(as.factor(df_feature))
  df_feature_numeric - 1
}

banking_dataset$HasDefault <- binary_factor_to_numeric(banking_dataset$HasDefault)
banking_dataset$HasHousingLoan <- binary_factor_to_numeric(banking_dataset$HasHousingLoan)
banking_dataset$HasPersonalLoan <- binary_factor_to_numeric(banking_dataset$HasPersonalLoan)
banking_dataset$HasSubscribedToDeposit <- binary_factor_to_numeric(banking_dataset$HasSubscribedToDeposit)

  # Head of data
head(banking_dataset)

  # Write unnormalized data to CSV
path_to_prepared_dataset = paste(getwd(), "/Data/banking-dataset-unnormalized.csv", sep="")
write.csv(banking_dataset, file=path_to_prepared_dataset, row.names=F)

  # Normalizing continuous variables
banking_dataset$Age <- scale(banking_dataset$Age)
banking_dataset$AnnualBalance <- scale(banking_dataset$AnnualBalance)
banking_dataset$CallDuration <- scale(banking_dataset$CallDuration)
banking_dataset$CurrentCampaignCalls <- scale(banking_dataset$CurrentCampaignCalls)
banking_dataset$PreviousCampaignsCalls <- scale(banking_dataset$PreviousCampaignsCalls)

  # Write normalized data to CSV
path_to_prepared_dataset = paste(getwd(), "/Data/banking-dataset-normalized.csv", sep="")
write.csv(banking_dataset, file=path_to_prepared_dataset, row.names=F)
