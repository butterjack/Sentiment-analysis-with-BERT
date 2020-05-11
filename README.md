# Sentiment-analysis-with-BERT
BERT stands for Bidirectional Encoder Representations from Transformers. 
Unlike recent language representation models BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.<br>
BERT is state-of-the-art natural language processing model from Google. Using its latent space, it can be repurpossed for various NLP tasks, such as sentiment analysis.
This simple wrapper based on Transformers and PyTorch achieves 83.5% accuracy on guessing positivity / negativity on tweets. <br>
The accuarcy of the model can be improved dramatically with providing the whole dataset to the model. We worked with a sample of the dataset to match the computation power and cost needed.

# Installation
You should install the vocabulary file, the configuration file and the weigths from https://www.kaggle.com/abhishek/bert-base-uncased to the inputs folder.

# How to use 

## prepare data
The dataset was extracted from the the sentiment140 dataset provided in kaggle. It contains 1,600,000 tweets extracted using the twitter api. You can download it from the link: https://www.kaggle.com/kazanova/sentiment140?fbclid=IwAR0bhTuv49t2LkRhNqaEfGhNdb-IOPq45ebCnBrB9xAQq5RHkowjd6M8xyw

## train and evaluate weights
Training with default parameters can be performed simply by. <br>
``` python train.py ``` <br>
Optionally, you can change certain parameters like batch and valid size and number of epochs from the config.py

## predict tweet
``` python app.py ``` <br>
A simple graphical user interface managed by tkinter will open to you with a textarea to write your tweet. 

