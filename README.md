# author_recognition_from_articles

## Description:

Experimenting with a dataset of 10,000 texts from Greek sites, we applied RNN methods in order to predict the similarity between texts and authors. We created a RNN model from scratch, we compined different hyper parameters in order to gain better results and we created a confusion matrx to visualize the results.

## Prerequisites:
Data and code were applied on Colab Notebook. 
### The modules need to be installed are:

F, nn, json, Counter, seaborn, pandas, pprint, tqdm, math, matplotlib, matplotlib.pyplot, torch, roc_auc_score, numpy, random, os, re, defaultdict, accuracy_score

## DATA:

Data are json formatted. The articles are technically a dictionary with texts grouped by authors' names like the following example:
{
	'Akis Panagiotou': ['This text is created by Akis Panagiotou',
		                'Some information about another in here'],
	'John Johnny': ['This text is created by John Johnny',
		                'Some information about another in here']
}

## Pipeline

After data collection, we manipulated them for better use in our RNN model. We built a Long Short Term Memory network (LSTM) witch is capable of learning long-term dependencies. We used different hyperparameters and we focused on the numbers of features that are created in the initialization phase. We tried the numbers of features to be 40, 50, 60, 70, 80, 90, 100, 150 and after we got the best learning curves we we created a confusion matrx to visualize the results.

## Authors
Akis Panagiotou, Dimitris Pappas

## Acknowledgments
I would like to express my gratitude and appreciation for Dimitris Pappas whose guidance, support and encouragement has been invaluable throughout this project.