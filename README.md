# language_detection

just playing around with language detection

## requirements and setup

this project was programmed in python 3.10.

this project relies on a cuda-enabled installation of `pytorch` 2.1 for training.

required packages are listed in `requirements.txt`, though i used `conda` for actual installation of some packages. 

## how-to

### prediction

you can run inference for a single sample, or run it in a "live" demo mode.

single example:

```
$ python langdet_detect.py --input "最新パンダのシャンシャン 中国での暮らしは？お相手は？"
Japanese
```

CLI interactive mode:

```
$ python langdet_detect.py --live

Transformer Language Classification Demo by Derek Homel
this model was trained on the WiLI-2018 dataset.
enter a 1~2 sentence string to detect its language.
enter 'q', 'quit' or 'exit' to exit.

query: Hij werd afgelopen zomer met hoongelach ontvangen. Want een metershoge ijsbeer die zijn behoefte doet in de gracht, daar sla je als stad toch een pleefiguur mee?
language: Dutch

query: 내년 상반기부터 마약 중독으로 치료보호 대상자 치료비에 건강보험이 적용된다.
language: Korean

query: Le prix du paquet de cigarettes passera à 12 euros en 2025 et 13 euros courant 2026, annonce le ministre de la Santé
language: French

query: q

thanks, bye!
```

### training

currently, only the [Wikipedia Language Identification](https://zenodo.org/records/841984) (WiLI-2018) dataset is supported.

to train, please download the zipped dataset from the above link and extract it. the default directory is `./language_detection/datasets/WiLi_2018`.

then, you may train with `langdet_train.py`. when done, it will save a TSV file of test set results to the checkpoint directory.

please run `python langdet_train.py --help` for s full list of arguments. 

the included sample model is trained with the default parameters.

### evaluation

if you want to only run the evaluation on the test set, you can use `langdet_test.py`.

this will load a pretrained checkpoint and run only test set inference, and save a TSV file.