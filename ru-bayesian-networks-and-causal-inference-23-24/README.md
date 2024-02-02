# Bayesian Networks and Causal Inference 2023/2024

This repository contains the fulfillment of the two assignments for the Radboud University course [NWI-IMC012 Bayesian Networks and Causal Inference](https://www.ru.nl/courseguides/science/vm/osirislinks/imc/nwi-imc012/) for the school year 2023-2024 as made by [Daan Brugmans](https://github.com/daanbrugmans) and Maarten Beerenschot.
The course's assignments consisted of two research projects to be performed by teams of students, one on the topic of Bayesian Networks and one on the topic of Causal Inference.

## Project 1: Bayesian Networks
[Project 1 Paper](BNCI_Assignment_1_Group_31.pdf)

Project 1 consisted of the construction of a Bayesian Network fitted on a dataset.
This dataset is the [Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) from the UCI ML repository.
We manually constructed a DAG (Directed Acyclic Graph) based on domain knowledge, ordering variables temporally, then tested the DAG using canonical correlations in order to find any missing or erroneous edges in the graph.
We aimed to find the variables most strongly causally correlated with the likelihood of a banking customer subscribing to a long-term deposit.
Our results showed that the successes of previous long-term deposit campaigns by the bank, the duration of a promotional call by the bank to the customer for such a previous campaign, and the job category of the customer were most highly causally correlated.

## Project 2: Causal Inference
[Project 2 Paper](BNCI_Assignment_2_Group_31.pdf)

Building on top of the findings of paper 1, we performed multiple causal inference analyses on the DAG between the likelihood of a banking customer subscribing to a long-term deposit and the successes of previous campaigns.
We performed covariate adjustment and instrumental variable analyses.

## Running the Code
In order to run the code, which may be found in the [code](code) folder, please make sure you have installed a recent version of [R](https://www.r-project.org/).
You must
Running the code in [Preparation.R](code/Preparation.R) produces some files required by the other code files, so it is advised to run it first.

