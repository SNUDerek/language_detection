# language_detection

a transformer encoder classifier for language identification.

this was done in partial completion of an application screening assignment.

if viewing offline, you can view the online version here:

[https://github.com/SNUDerek/language_detection](https://github.com/SNUDerek/language_detection)

## about

this is an example of a basic transformer encoder (BERT-style) classifier trained on the language identification (LID) task.

the obvious solution would be to take a pretrained multilingual encoder and fine-tune it on my dataset (*and the smart alec way would be to make a function that calls GPT-4 with a prompt asking to identify a language*), but i thought that was against the spirit of the assignment, so i implemented and trained my own solution according to my own self-imposed guidelines:

- i wanted the implementation to be non-trivial and educational
    - trained on WiLI-2018, a large dataset with 235,000 total paragraphs in 235 languages
    - decided to use modern transformer-based architecture
    - implemented the model and training loop in `pytorch` without external libraries (`huggingface` etc.)
    - as a development challenge, used newer python and tried to keep code relatively organized and typed, with usability in mind

- i wanted the solution to be unique and experiment with my own ideas
    - no pretrained models used, and no pre-training on external datasets
    - no subword tokenization; input is unicode byte sequence as integers `0~256` + control tokens
    - due to time constraints and curiosity, model is trained on classification and masked LM objectives jointly 

### results

these results are from the included checkpoint `./experiments/wili2018/wili2018-checkpoint-000020.pt`  
(*github version has no checkpoints included*)

```
classification results, test set:

macro precision: 0.91144
macro recall:    0.90686
macro F1:        0.90768
```

see below for best- and worst-performance languages, and see the `analysis.ipynb` for test set macro F1 score and class F1 score statistics, along with some error analysis.

### references

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805), Devlin, *et al.* 2018

from BERT, i adopted the basic transformer encoder classification architecture

[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) Liu, *et al.* 2019

like RoBERTa, i used dynamic masks for the masked LM objective, dropped next-sentence prediction, and trained on long sequences

[Byte-based Multilingual NMT for Endangered Languages](https://aclanthology.org/2022.coling-1.388/) Zhang and Xu, 2022

there is a large body of using byte-based inputs, but i first saw the idea in this paper

## requirements

this project was programmed in python 3.10. it relies on a cuda-enabled installation of `pytorch` 2.1 for training.

required packages are listed in `requirements.txt`, though i used `conda` for actual installation of some packages. 

## how-to

### prediction

you can run inference for a single sample, or run it in a "live" demo mode.

you can specify a trained checkpoint to load with `--checkpoint_file`; offline version includes a default checkpoint.

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

you can check `analysis.ipynb` for a brief error analysis of the data.

## best and worst performing languages

from `analysis.ipynb`

```
ten languages by highest F1 score:

lang: nan              nan     	F1: 0.99501, prc: 0.99205, rcl: 0.99800
lang: Maori            mri     	F1: 0.99599, prc: 0.99799, rcl: 0.99400
lang: Malagasy         mlg     	F1: 0.99600, prc: 0.99600, rcl: 0.99600
lang: Uighur           uig     	F1: 0.99699, prc: 1.00000, rcl: 0.99400
lang: Lojban           jbo     	F1: 0.99800, prc: 0.99602, rcl: 1.00000
lang: Burmese          mya     	F1: 0.99800, prc: 1.00000, rcl: 0.99600
lang: Tibetan          bod     	F1: 0.99900, prc: 1.00000, rcl: 0.99800
lang: Dhivehi          div     	F1: 0.99900, prc: 1.00000, rcl: 0.99800
lang: Central Kurdish  ckb     	F1: 1.00000, prc: 1.00000, rcl: 1.00000
lang: Navajo           nav     	F1: 1.00000, prc: 1.00000, rcl: 1.00000
```

```
ten languages by lowest F1 score:

lang: Croatian         hrv     	F1: 0.44250, prc: 0.43156, rcl: 0.45400
lang: Bosnian          bos     	F1: 0.44815, prc: 0.55457, rcl: 0.37600
lang: Pampanga         pam     	F1: 0.46772, prc: 0.82741, rcl: 0.32600
lang: Serbo-Croatian   hbs     	F1: 0.47122, prc: 0.42810, rcl: 0.52400
lang: Indonesian       ind     	F1: 0.48993, prc: 0.55584, rcl: 0.43800
lang: English          eng     	F1: 0.54071, prc: 0.43890, rcl: 0.70400
lang: Malay            msa     	F1: 0.56330, prc: 0.55299, rcl: 0.57400
lang: German           deu     	F1: 0.60388, prc: 0.47138, rcl: 0.84000
lang: Banyumasan       map-bms 	F1: 0.61326, prc: 0.53858, rcl: 0.71200
lang: Chavacano        cbk     	F1: 0.66294, prc: 0.62021, rcl: 0.71200
```

a qualitative analysis of false-negatives and false-positives for English and German show that there appear to be a number of mis-tagged elements in the test set. please see the notebook for details.

## future work

there are many things i'd like to do to enhance the project if given more time:

- unit tests
- support other popular language identification datasets
- compare my bytes input approach to the alternatives i proposed above (e.g. fine-tuning a pretrained model)
- track training loss with integration with something like weights & biases or tensorboardx
- comparison of model hyperparameter settings, scripts for hyperparameter tuning for model architecture
- more error analysis
- better inference methods, such as web-based inference server