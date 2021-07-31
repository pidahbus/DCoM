[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dcom-a-deep-column-mapper-for-semantic-data/semanticity-prediction-on-viznet)](https://paperswithcode.com/sota/semanticity-prediction-on-viznet?p=dcom-a-deep-column-mapper-for-semantic-data)

# DCoM
**D**eep **Co**lumn **M**apper is a generalized deep learning framework which classifies the type of a column given its values. Under this framework there are several models at work. This repository allows to train, finetune and predict with a DCoM model. The academic paper link will be attached soon.

## Data
Sample data has been provided in the data folder. The sample data is split in train, valid and test. Each CSV contains two columns, type of the column and column values. To get the access of the data contact the authors of [Sherlock: A Deep Learning Approach to Semantic Data Type Detection](https://arxiv.org/abs/1905.10688).

## Model Training
Model training is very straightforward. Change the parameters in [config.py](./config.py) and run in the terminal,
```
pip install -r requirements.txt
python train.py
```

## Model Finetuning
Set the path of a pretrained model and metadata in INITIAL_MODEL_WEIGHT_PATH and INITIAL_METADATA_PATH respectively in [config.py](./config.py) to finetune the model from the given checkpoint.

## Model Predictions
Sample prediction of orientation angles given images has been shown in [notebook.ipynb](./notebook.ipynb). 

## Results
Using a novel data preparation technique and NLP models, DCoM outperforms other works in this domain. The table compares the performace of DCoM with others. 


| Method        | Engineered Features |   k   | F1 Score | Runtime (s) | Size (MB) |
| ------------- | :-----------------: | :---: | :------: | :---------: | :-------: |
|DCoM-Single-LSTM | Yes | 1 | 0.895 | 0.019 | 112.1 |
|DCoM-Single-LSTM | Yes | 10 | 0.898 | 0.152 | 112.1|
|DCoM-Single-LSTM | No | 1 | 0.871 | 0.018 | 97.8|
|DCoM-Single-LSTM | No | 10 | 0.877 | 0.141 | 97.8|
|DCoM-Multi-LSTM | Yes | 1 | 0.878 | 0.046 | 4.7|
|DCoM-Multi-LSTM | Yes | 10 | 0.881 | 0.416 | 4.7|
|DCoM-Multi-LSTM | No | 1 | 0.869 | 0.044 | 4.6 |
|DCoM-Multi-LSTM | No | 10 | 0.871 | 0.401 | 4.6|
|DCoM-Single-DistilBERT | Yes | 1 | 0.922 | 0.162 | 268.2|
|DCoM-Single-DistilBERT | Yes | 10 | 0.925 | 1.552 | 268.2|
|DCoM-Single-DistilBERT | No | 1 | 0.901 | 0.158 | 202.3|
|DCoM-Single-DistilBERT | No | 10 | 0.904 | 1.492 | 202.3|
|DCoM-Single-Electra | Yes | 1 | 0.907 | 0.093 | 53.1|
|DCoM-Single-Electra | Yes | 10 | 0.909 | 0.894 | 53.1|
|DCoM-Single-Electra | No | 1 | 0.890 | 0.092 | 45.7|
|DCoM-Single-Electra | No | 10 | 0.892 | 0.887 | 45.7|
|Sherlock| - | - | 0.890 | 0.42 | 6.2 |
