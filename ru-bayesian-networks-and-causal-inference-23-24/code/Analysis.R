rm(list=ls())
setwd("C:/Users/Daan/Documents/Projecten/ru-bayesian-networks-and-causal-inference-23-24") # Repository root

library(ggplot2)

path_to_dataset = paste(getwd(), "/data/banking-dataset-unnormalized.csv", sep="")
banking_dataset <- read.csv(path_to_dataset, sep=",", stringsAsFactors=FALSE)

  # Data Visualization
banking_dataset$HasSubscribedToDeposit <- as.factor(banking_dataset$HasSubscribedToDeposit)

summary(banking_dataset$Age)
ggplot(banking_dataset, aes(x=Age)) +
  geom_density()
ggplot(banking_dataset, aes(x=Age, fill=HasSubscribedToDeposit)) +
  geom_boxplot()

summary(banking_dataset$AnnualBalance)
ggplot(banking_dataset, aes(x=AnnualBalance, fill=HasSubscribedToDeposit)) +
  geom_density(alpha=0.5)
ggplot(banking_dataset, aes(x=AnnualBalance , fill=HasSubscribedToDeposit)) +
  geom_boxplot()

summary(banking_dataset$CallDuration)
ggplot(banking_dataset, aes(x=CallDuration)) +
  geom_density()
ggplot(banking_dataset, aes(x=CallDuration, fill=HasSubscribedToDeposit)) +
  geom_boxplot()

summary(banking_dataset$CurrentCampaignCalls)
ggplot(banking_dataset, aes(x=CurrentCampaignCalls, fill=HasSubscribedToDeposit)) +
  geom_bar() +
  facet_wrap(~HasSubscribedToDeposit)

ggplot(banking_dataset, aes(x=EducationLevel, fill=EducationLevel)) +
  geom_bar() +
  facet_wrap(~HasSubscribedToDeposit)

ggplot(banking_dataset, aes(x=HasDefault, fill=HasDefault)) +
  geom_bar() +
  facet_wrap(~HasSubscribedToDeposit)

ggplot(banking_dataset, aes(x=HasHousingLoan, fill=HasHousingLoan)) +
  geom_bar() +
  facet_wrap(~HasSubscribedToDeposit)

ggplot(banking_dataset, aes(x=HasPersonalLoan, fill=HasPersonalLoan)) +
  geom_bar() +
  facet_wrap(~HasSubscribedToDeposit)

ggplot(banking_dataset, aes(x=JobCategory, fill=JobCategory)) +
  geom_bar() +
  facet_wrap(~HasSubscribedToDeposit)

ggplot(banking_dataset, aes(x=MaritalStatus, fill=MaritalStatus)) +
  geom_bar() +
  facet_wrap(~HasSubscribedToDeposit)

summary(banking_dataset$PreviousCampaignsCalls)
ggplot(banking_dataset, aes(x=PreviousCampaignsCalls)) +
  geom_density()
ggplot(banking_dataset, aes(x=PreviousCampaignsCalls, fill=HasSubscribedToDeposit)) +
  geom_boxplot()

ggplot(banking_dataset, aes(x=PreviousCampaignOutcome, fill=PreviousCampaignOutcome)) +
  geom_bar() +
  facet_wrap(~HasSubscribedToDeposit)

ggplot(banking_dataset, aes(x=HasSubscribedToDeposit, fill=HasSubscribedToDeposit)) +
  geom_bar()
