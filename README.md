# language_detection

just playing around with language detection

## requirements and setup

this project was programmed in python 3.10.

this project uses a cuda-enabled installation of `pytorch` 2.1.

other required packages are listed in `requirements.txt`, though i used `conda` for actual installation of some packages. 

## description

this uses a BERT-inspired architecture to language detection from unicode byte sequences:

- jointly train with masked language model and classification objectives
- no next-sentence-prediction
- MLM by default uses the BERT 15% rate, with BERT defaults (80/10/10 mask/replace/keep) 

## api

### training

TODO

### prediction

TODO