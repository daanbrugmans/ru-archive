# Computational Psycholinguistics 2023/2024

This repository contains the completed assignments for the Radboud University course [LET-REMA-LCEX28 Computational Psycholinguistics](https://www.ru.nl/courseguides/arts/courses/ma/rema-lc/let-rema-lcex28/) for the school year 2023-2024 as made by [Daan Brugmans](https://github.com/daanbrugmans).
The Computational Psycholinguistics course is about the application of computational methods and models within psycholinguistics research.
In addition to a weekly paper that must be read as part of a reading list, students of the course must fulfill two assignments in which they conduct their own computational psycholinguistics research on a small scale.

## Assignment 1
[Assignment 1 Paper](/assignment-1/report.pdf)

For assignment 1, students train a set of Word2Vec models on a given dataset.
This dataset contains sentences read by participants of a study where the participant must determine as quickly as possible whether the target word of the sentence is a real word or not.
The participants' response times were measured and included in the dataset.
Given this data, students must train the Word2Vec models on the sentences and compare the models' predictions to the measured response times, specifically the response times for the target word, and make a claim on how well these response times are predicted by the models' predictions.

## Assignment 2
[Assignment 2 Paper](/assignment-2/report/report.pdf)

For assignment 2, students must train a set of Recurrent Neural Networks (RNNs) on a self-chosen dataset of an experiment in which participants' ERP components are measured while reading.
This assignment was based on the then recent discovery that the P600 ERP component may be the backpropagation of language error in the human brain, just as gradients are the backpropagation of language error in RNNs.
By training RNNs on the sentence data, the RNNs' gradients may be compared to the participants' P600 components, as is part of the assignment so that students may make a claim on how well these two correlate.
