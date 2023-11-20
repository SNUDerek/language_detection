# language_detection

just playing around with language detection

## requirements and setup

this project was programmed in python 3.10.11 and uses `pipenv` to manage packages.

after installing with `pip install pipenv` you may activate shell with `pipenv shell` or run a script with `pipenv run <script>`

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
    'labels': dict[str | int, str],
    'dropped': list[str] | None}
```