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
│   ├── preprocessing.py
│   └── sentences_classification_preprocessing.py
├── dataset
│   └── imdb
│       ├── imdb.neg
│       └── imdb.pos
├── evaluations
│   ├── __init__.py
│   └── metrics.py
├── examples
│   ├── __init__.py
│   └── sentence_classification
│       ├── __init__.py
│       └── cbow_cnn_example.py
├── net
│   ├── __init__.py
│   ├── cbow.py
│   └── cnn.py
├── saved_models
│   └── sentence_classification.pt
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
    ├── losses.py
    ├── optimizer.py
    ├── save_load_models.py
    └── trainer.py

```
# Tasks

- Sentence Classification

# Credits

[Yihan Zhong](https://github.com/YIHAN-ZHONG)
[Marwan Mashra](https://github.com/MarwanMashra)