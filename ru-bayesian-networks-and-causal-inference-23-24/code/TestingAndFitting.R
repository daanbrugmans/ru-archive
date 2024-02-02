rm(list=ls())
setwd("C:/Users/Daan/Documents/Projecten/ru-bayesian-networks-and-causal-inference-23-24")
options(digits=3)

remotes::install_github("jtextor/dagitty/r")

library(tidyverse)
library(dagitty)
library(fastDummies)
library(CCP)

path_to_dataset = paste(getwd(), "/data/banking-dataset-normalized.csv", sep="")
banking_dataset <- read.csv(path_to_dataset, sep=",", stringsAsFactors=T)

dag_initial <- dagitty('dag {
Age [pos="-1.441,-1.094"]
AnnualBalance [pos="-1.269,-0.713"]
CurrentCampaignCalls [pos="-0.824,-0.388"]
HasDefault [pos="-1.198,-0.597"]
CallDuration [pos="-0.935,-0.664"]
EducationLevel [pos="-1.077,-1.102"]
HasHousingLoan [pos="-1.412,-0.549"]
JobCategory [pos="-1.198,-0.995"]
HasPersonalLoan [pos="-1.002,-0.776"]
MaritalStatus [pos="-1.351,-0.791"]
PreviousCampaignOutcome [pos="-1.226,-0.236"]
PreviousCampaignsCalls [pos="-0.983,-0.232"]
HasSubscribedToDeposit [outcome,pos="-1.231,-0.403"]
Age -> EducationLevel
Age -> JobCategory
Age -> MaritalStatus
AnnualBalance -> HasDefault
AnnualBalance -> HasHousingLoan
AnnualBalance -> HasPersonalLoan
AnnualBalance -> HasSubscribedToDeposit
CurrentCampaignCalls -> CallDuration
CurrentCampaignCalls -> HasSubscribedToDeposit
HasDefault -> HasPersonalLoan
HasDefault -> HasSubscribedToDeposit
CallDuration -> HasSubscribedToDeposit
EducationLevel-> JobCategory
HasHousingLoan -> HasDefault
HasHousingLoan -> HasSubscribedToDeposit
JobCategory -> AnnualBalance
HasPersonalLoan -> HasSubscribedToDeposit
MaritalStatus -> AnnualBalance
PreviousCampaignOutcome -> HasSubscribedToDeposit
PreviousCampaignsCalls -> CurrentCampaignCalls
PreviousCampaignsCalls -> PreviousCampaignOutcome
}
')

plot(dag_initial)

independency_tests <- localTests(dag_initial, banking_dataset, type="cis.pillai")
plotLocalTestResults(independency_tests)
path_to_local_tests_csv = paste(getwd(), "/Data/cis-pillai-test-results.csv", sep="")
write.csv(format(independency_tests[, 0:2], digits=2), file=path_to_local_tests_csv, row.names=T)

get_canonical_correlations <- function(dag, dataset) {
  canonical_correlations <- c()

  for(variable_name in names(dag)) {
    for(parent in parents(dag, variable_name)) {
      other_parents <- setdiff(parents(dag, variable_name), parent)
      tst <- ciTest(
        X=variable_name,
        Y=parent,
        Z=other_parents,
        banking_dataset,
        type="cis.pillai"
      )

      canonical_correlations <- rbind(
        canonical_correlations,
        data.frame(
          list(
            X=parent,
            A="->",
            Y=variable_name,
            cor=tst[,"estimate"],
            p=tst[,"p.value"]
          )
        )
      )
    }
  }

  canonical_correlations
}

get_dag_with_canonical_correlations <- function(dag, canonical_correlations) {
  cancor_edge_coefficients <- paste(
    canonical_correlations$X,
    canonical_correlations$A,
    canonical_correlations$Y,
    "[beta=",signif(canonical_correlations$cor,2),"] ",
    collapse="\n"
  )

  dag_with_coefficients <- dagitty(cancor_edge_coefficients)
  coordinates(dag_with_coefficients) <- coordinates(dag)

  dag_with_coefficients
}

canonical_correlations <- get_canonical_correlations(dag_initial, banking_dataset)
dag_initial_with_cancor <- get_dag_with_canonical_correlations(dag_initial, canonical_correlations)
plot(dag_initial_with_cancor, show.coefficients=T)

  # Altering the initial DAG until the desired DAG is reached
dag_altered <- dagitty('dag {
Age [pos="0,-2"]
AnnualBalance [pos="-.3,-1.65"]
CurrentCampaignCalls [pos="-.35,-1"]
HasDefault [pos="-0.45,-1.5"]
CallDuration [pos="-.35,-.5"]
EducationLevel [pos="-0.15, -2"]
HasHousingLoan [pos="-0.140,-1.220"]
JobCategory [pos="0,-1.5"]
HasPersonalLoan [pos="-0.3,-1.3"]
MaritalStatus [pos="-0.2,-1.65"]
PreviousCampaignOutcome [pos="-.15,-.75"]
PreviousCampaignsCalls [pos="-.25,-1.15"]
HasSubscribedToDeposit [outcome,pos="0,-.5"]
Age -> AnnualBalance
Age -> EducationLevel
Age -> JobCategory
Age -> HasHousingLoan
Age -> MaritalStatus
AnnualBalance -> HasDefault
AnnualBalance -> HasPersonalLoan
CurrentCampaignCalls -> CallDuration
HasDefault -> HasPersonalLoan
CallDuration -> HasSubscribedToDeposit
EducationLevel -> AnnualBalance
EducationLevel -> JobCategory
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

independency_tests_alt <- localTests(dag_altered, banking_dataset, type="cis.pillai")
independency_tests_alt <- independency_tests_alt %>%
  filter(estimate <= -0.06 | estimate >= 0.06)
independency_tests_alt

canonical_correlations_alt <- get_canonical_correlations(dag_altered, banking_dataset)
dag_altered_with_cancor <- get_dag_with_canonical_correlations(dag_altered, canonical_correlations_alt)
filter(canonical_correlations_alt, cor >= -0.06 & cor <= 0.06)
plot(dag_altered_with_cancor, show.coefficients=T)
