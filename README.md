# language_detection

a transformer encoder classifier for language identification.

## about

this is an example of a basic transformer encoder (BERT-style) classifier trained on the language identification (LID) task.

- i wanted the implementation to be non-trivial and educational
    - trained on WiLI-2018, a large dataset with 235,000 total paragraphs in 235 languages
    - decided to use modern transformer-based architecture
    - implemented the model and training loop in `pytorch` without external libraries (`huggingface` etc.)
    - as a development challenge, used newer python and tried to keep code relatively organized and typed, with usability in mind

- i wanted the solution to be unique and experiment with my own ideas
    - no pretrained models used, and no pre-training on external datasets
    - no subword tokenization; input is unicode byte sequence as integers `0~256` + control tokens
    - due to time constraints and curiosity, model is trained on classification and masked LM objectives jointly 

### references

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805), Devlin, *et al.* 2018

from BERT, i adopted the basic transformer encoder classification architecture

[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) Liu, *et al.* 2019

like RoBERTa, i used dynamic masks for the masked LM objective, dropped next-sentence prediction, and trained on long sequences

[Byte-based Multilingual NMT for Endangered Languages](https://aclanthology.org/2022.coling-1.388/) Zhang and Xu, 2022

i'm sure there is a large body of using byte-based inputs, but i first saw the idea in this paper

## requirements

this project was programmed in python 3.10. it relies on a cuda-enabled installation of `pytorch` 2.1 for training.

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

## future work

there are many things i'd like to do to enhance the project:

- support other popular language identification datasets
- track training loss with integration with something like weights & biases or tensorboardx
- comparison of model hyperparameter settings, scripts for hyperparameter tuning for model architecture
- more error analysis
- better inference methods, such as web-based inference server