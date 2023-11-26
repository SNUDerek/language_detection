# language_detection

just playing around with language detection

## requirements and setup

this project was programmed in python 3.10.

this project uses cuda-enabled installation of `pytorch` 2.1.

other required packages are listed in `requirements.txt`, though i used `conda` for actual installation of some packages. 

## description

BERT based classification

- jointly train MLM and classifier
- MLM by default uses the BERT 15% rate, with BERT defaults (80/10/10 mask/replace/keep) 
- drop the next-sentence-prediction (roberta?)

## api

### data loading

you can load the [Wikipedia Language Identification database](https://zenodo.org/records/841984) after downloading and extracting the zip file.

by default, `dropped_duplicates` is `True`. this will drop any training samples that also appear in the test set (3117 total).

```
from language_detection.data import load_wili_2018_dataset

wili_2018_data_path = "/path/to/extracted/wili_dataset"

wiki_dataset = load_wili_2018_dataset(wili_2018_data_path)
```

loading functions return a `RawDataset` object:

```
print(wiki_dataset.__annotations__)

>>> {'x_train': list[str],
    'x_test': list[str],
    'y_train': list[str],
    'y_test': list[str],
    'idx2lang': dict[int, str],
    'lang2idx': dict[str, int],
    'labels': dict[str, str] | None,
    'dropped': list[str] | None}
```