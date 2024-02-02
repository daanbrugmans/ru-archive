# Information Retrieval 2023/2024

This repository contains the fulfillment of the assignments of the Radboud University course [NWI-I00041 Information Retrieval](https://www.ru.nl/courseguides/science/vm/osirislinks/i00/nwi-i00041/) for the school year 2023-2024 as made by [Daan Brugmans](https://github.com/daanbrugmans).

In addition, this repository also contains a fork of the realization of [the project for the course](https://github.com/JulianRodd/projectInformationRetrieval). This was done in collaboration with [Janneke Nouwen](https://github.com/JannekeNouwen) and [Julian Roddeman](https://github.com/JulianRodd).

## Project
[Information Retrieval Project paper](projectInformationRetrieval/paper/Research.pdf)

The [projectInformationRetrieval folder](projectInformationRetrieval) contains all relevant files for the project of the course.
The goal of the project is for students to put their knowledge of IR into practice.
For our realization of the project, we chose to study the adaptation of Neural IR models in unseen domains.
Specifically, we studied the change in performance of a BERT-based retrieval model when ranking document-query pairs in the COVID-19 domain.
We compared the model's rankings before and after fine-tuning it on a part of the TREC-COVID dataset using Parameter-Efficient Fine-Tuning (PEFT).
Our results showed that rankings improved after applying PEFT, indicating that PEFT may be a relatively low-resource way to adapt large neural rankers on new domains.

## Assignments
The assignments were performed individually and tested the students' capability to understand the core of the Information Retrieval course: the indexing of a document collection, the retrieval of documents relevant to a query, and the evaluation of retrieval performance.