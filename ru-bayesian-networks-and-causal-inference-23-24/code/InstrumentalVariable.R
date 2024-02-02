rm(list=ls())
setwd("C:/Users/Daan/Documents/Projecten/ru-bayesian-networks-and-causal-inference-23-24") # Repository root

library(dagitty)
library(ggplot2)

  # Load banking dataset
path_to_dataset = paste(getwd(), "/data/banking-dataset-normalized.csv", sep="")
banking_dataset <- read.csv(path_to_dataset, sep=",", stringsAsFactors=T)

  # Remove outliers of candidate IV
#ggplot(banking_dataset, aes(x=PreviousCampaignsCalls)) + geom_boxplot()
#banking_dataset <- banking_dataset[banking_dataset$PreviousCampaignsCalls < 100,] # Major outlier removed
#banking_dataset <- banking_dataset[banking_dataset$PreviousCampaignsCalls < 17.5,]
#banking_dataset <- banking_dataset[banking_dataset$PreviousCampaignsCalls < 12.5,]
#banking_dataset <- banking_dataset[banking_dataset$PreviousCampaignsCalls < 0,] #No outliers

  # Load altered version of DAG
dag_altered <- dagitty('dag {
Age [pos="-0.642,-3.240"]
AnnualBalance [pos="-3.721,-3.257"]
CallDuration [pos="-0.597,2.835"]
CurrentCampaignCalls [pos="2.497,2.835"]
EducationLevel [pos="-2.211,-2.203"]
HasDefault [pos="-5.380,-2.121"]
HasHousingLoan [pos="2.497,-1.067"]
HasPersonalLoan [pos="-3.736,-1.083"]
HasSubscribedToDeposit [outcome,pos="-0.627,1.090"]
JobCategory [pos="-0.642,-1.067"]
MaritalStatus [pos="2.497,-3.208"]
PreviousCampaignOutcome [exposure,pos="2.497,1.090"]
PreviousCampaignsCalls [pos="4.306,-0.359"]
Age -> AnnualBalance
Age -> EducationLevel
Age -> HasHousingLoan
Age -> JobCategory
Age -> MaritalStatus
AnnualBalance -> HasDefault
AnnualBalance -> HasPersonalLoan
CallDuration -> HasSubscribedToDeposit
CurrentCampaignCalls -> CallDuration
EducationLevel -> AnnualBalance
EducationLevel -> JobCategory
HasDefault -> HasPersonalLoan
HasHousingLoan -> HasSubscribedToDeposit
JobCategory -> HasHousingLoan
JobCategory -> HasPersonalLoan
JobCategory -> HasSubscribedToDeposit
JobCategory -> MaritalStatus
JobCategory -> PreviousCampaignOutcome
MaritalStatus -> HasHousingLoan
PreviousCampaignOutcome -> CurrentCampaignCalls
PreviousCampaignOutcome -> HasHousingLoan
PreviousCampaignOutcome -> HasSubscribedToDeposit
PreviousCampaignsCalls -> PreviousCampaignOutcome
}
')
plot(dag_altered)

  # Identify instrumental variables in DAG
exposures(dag_altered) <- "PreviousCampaignOutcome"
outcomes(dag_altered) <- "HasSubscribedToDeposit"
instrumentalVariables(dag_altered) # IV found: PreviousCampaignsCalls

  # Calculate unbiased regression line for IV by d-separating on other paths from IV to outcome
    # The IV must only influence the outcome through the exposure.
    # That is, IV -> exposure -> (variables) -> outcome must be the only paths through which IV can reach outcome.
      # Allowed/Include:
        # IV -> exposure -> outcome
        # IV -> exposure -> variable* -> outcome
      # Disallowed/Exclude:
        # IV -> variable* -> outcome
        # IV -> variable* -> exposure -> outcome (this is a conditional IV)
    # This is done by calculating the regression line of the exposure given the IV and variables to be d-separated.
    # This adjusts the exposure to the d-separated variables.
    # The predictions of this regression are used as a replacement for the original exposure.variable.
nonconditional.iv.estimate.regression <- function(outcome, exposure, IV, dataset, debias=T) {
  if (debias) {
    adjusted_exposure_regression <- lm(dataset[[exposure]] ~ dataset[[IV]], dataset)
    adjusted_exposure <- predict(adjusted_exposure_regression)
  } else {
    adjusted_exposure <- dataset[[exposure]]
  }

  lm(dataset[[outcome]] ~ adjusted_exposure, dataset)
}

  # Calculate unbiased IV estimate
unbiased_outcome_regression <- nonconditional.iv.estimate.regression("HasSubscribedToDeposit", "PreviousCampaignOutcome", "PreviousCampaignsCalls", banking_dataset)
plot(unbiased_outcome_regression)

iv_estimate <- coef(unbiased_outcome_regression)[2]
iv_estimate

iv_estimate_confidence_interval <- confint(unbiased_outcome_regression)
iv_estimate_confidence_interval

  # Calculate biased IV estimate
biased_outcome_regression <- nonconditional.iv.estimate.regression("HasSubscribedToDeposit", "PreviousCampaignOutcome", "PreviousCampaignsCalls", banking_dataset, debias=F)
biased_outcome_regression

biased_iv_estimate_confidence_interval <- confint(biased_outcome_regression)
biased_iv_estimate_confidence_interval
