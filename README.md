# Some classic tasks and network in NLP

# Instructions

## Enviroment
we used
```
python version：3.8
```

## Dependencies
you can install all the dependencies by
```
pip install -r requirements.txt
```


# Code structure
``` 
├── README.md
├── config.json
├── config.py
├── data_processing
│   ├── __init__.py
│   ├── data_loader.py
│   ├── language_modeling_preprocessing.py
│   ├── preprocessing.py
│   ├── sentences_classification_preprocessing.py
│   └── word_dict.py
├── dataset
│   ├── imdb
│   │   ├── imdb.neg
│   │   └── imdb.pos
│   └── ptb
│       ├── README
│       ├── ptb.char.test.txt
│       ├── ptb.char.train.txt
│       ├── ptb.char.valid.txt
│       ├── ptb.test.txt
│       ├── ptb.train.txt
│       └── ptb.valid.txt
├── evaluations
│   ├── __init__.py
│   ├── metrics.py
│   └── perplexity.py
├── examples
│   ├── __init__.py
│   ├── language_models
│   │   ├── __init__.py
│   │   ├── lstm_lm_example.py
│   │   └── n_gram_lm_example.py
│   └── sentence_classification
│       ├── __init__.py
│       └── cbow_cnn_example.py
├── ipynb_scripts
├── language_models
│   ├── __init__.py
│   └── models
│       ├── __init__.py
│       ├── lstm_lm.py
│       └── n_grams_lm.py
├── net
│   ├── __init__.py
│   ├── cbow.py
│   ├── cnn.py
│   ├── lstm.py
│   └── n_gram.py
├── requirements.txt
├── saved_models
├── sentence_classification
│   ├── __init__.py
│   ├── models
│   │   ├── __init__.py
│   │   └── cbow_cnn.py
│   └── training_scripts
│       ├── __init__.py
│       └── training_cbow_cnn.py
├── test
│   ├── __init__.py
│   └── test_sentences_classification_preprocessing.py
└── utils
    ├── __init__.py
    ├── dropout.py
    ├── losses.py
    ├── optimizer.py
    ├── save_load_models.py
    ├── trainer.py
    ├── trainer_cbow_cnn.py


```
# Tasks

- Sentence Classification
- Language models

# Credits

[Yihan Zhong](https://github.com/YIHAN-ZHONG)

[Marwan Mashra](https://github.com/MarwanMashra)