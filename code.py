import torch
import torch.nn.functional as F
import torch.nn as nn
import json
from collections import Counter
import seaborn as sns
import pandas as pd
from pprint import pprint
%matplotlib inline
from tqdm import tqdm
import math
import matplotlib
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import os
import re
from collections import defaultdict
from sklearn.metrics import accuracy_score

from google.colab import drive
drive.mount('/content/drive/', force_remount=True)


with open('/content/drive/My Drive/data_for_colab/paper/last_data2/train200.json') as f:
    train = json.load(f)
with open('/content/drive/My Drive/data_for_colab/paper/last_data2/dev200.json') as f:
    dev = json.load(f)
with open('/content/drive/My Drive/data_for_colab/paper/last_data2/test200.json') as f:
    test = json.load(f)


def delete_strange_authors(mytrain, mydev, mytest):
  removedWords = ['Του ','Της ','Των ','Υπεύθυνος: ','ΕΠΙΜΕΛΕΙΑ: ', 'Επιμέλεια: ','Από ', 'τον ', 'τη ', 'Με ','ΚΕΙΜΕΝΑ -  ','ΚΡΙΤΙΚΗ:  ','ΚΡΙΤΙΚΗ: ']
  mytrain = {k:v for k,v in mytrain.items() if ('@' not in k and '.' not in k)}
  mytrain = {k:v for k,v in sorted(mytrain.items())}
  mydev = {k:v for k,v in mydev.items() if ('@' not in k and '.' not in k)}
  mydev = {k:v for k,v in sorted(mydev.items())}
  mytest = {k:v for k,v in mytest.items() if ('@' not in k and '.' not in k)}
  mytest = {k:v for k,v in sorted(mytest.items())}
  newTrain = {}
  newDev   = {}
  newTest  = {}
  for item1, item2, item3 in zip(mytrain.items(), mydev.items(), mytest.items()):
    new = item1[0]
    for w in removedWords:
      if w in new:
        new = new.replace(w,'')
    if new not in newTrain.keys():
      newTrain.update({new:item1[1]})
      newDev.update({new:item2[1]})
      newTest.update({new:item3[1]})
  k = ['Θεσσαλονίκη','ΔΝΚ','Αθήνα','ΑΝΤΩΝΗ ΓΑΛΑΝΟΠΟΥΛΟΥ','ΝΙΝΟΣ ΦΕΝΕΚ  ΜΙΚΕΛΙΔΗΣ','ΛΗΔΑΣ ΠΑΠΑΔΟΠΟΥΛΟΥ','ΧΡΥΣΟΥΛΑΣ ΠΑΠΑΪΩΑΝΝΟΥ','Μγς','ΠΑΣ',' ΓΙΑΝΝΗΣ ΤΡΙΑΝΤΗΣ','ΜΑΝΤΑΙΟΣ']
  for item in k:
    if item in list(newTrain):
      del newTrain[item]
      del newDev[item]
      del newTest[item]

  return newTrain, newDev, newTest

def choose_size(train, dev, test, fraction, min_authors = 50):
  train_texts = int (160*fraction)
  dev_texts   = int (20*fraction)
  res1 = {}
  for key, values in train.items():
      res1.update({key: values[0:train_texts]})
  res2 = {}
  for key, values in dev.items():
      res2.update({key: values[0:dev_texts]})
  res3 = {}
  for key, values in test.items():
      res3.update({key: values[0:dev_texts]})
  train = dict(list(res1.items())[0:min_authors])
  dev = dict(list(res2.items())[0:min_authors])
  test = dict(list(res3.items())[0:min_authors])
  return train, dev, test


def getVIA(mytrain):
  wordsPerText = Counter()
  for values in mytrain.values():
    for value in values:
      b = get_words(value)
      if len(b)>200:     #  use this for MLP
          b = b[0:200]
      wordsPerText.update(Counter(set(b)))
  wordsPerText = dict(wordsPerText)
  wordsPerText = {k:v for k,v in sorted(wordsPerText.items())}
  vocabulary = [x for x,y in wordsPerText.items() if y>5]
  vocabulary.insert(0,'PAD')
  idf = {x:(1.0/y) for x,y in wordsPerText.items() if y>5}
  authorsNames = [x for x in mytrain.keys()]
  return vocabulary, authorsNames, idf



# Functions for Text Manipulation

def replace_all(text):
  dic = {'ά': 'α', 'έ': 'ε', 'ί': 'ι', 'ϊ': 'ι', 'ΐ': 'ι', 'ή': 'η', 'ύ': 'υ', 'ϋ': 'υ', 'ΰ': 'υ', 'ώ': 'ω', 'ό': 'ο'}
  for i, j in dic.items():
    text = text.replace(i, j)
  return text

def get_words(s):
  w = s.lower()
  w = replace_all(w)
  w = re.sub('[^A-Za-zΑ-Ωα-ω]+', ' ', w)
  w = w.split()
  # w = [word for word in w if (any(i.isdigit() for i in word) == False) and (word not in stopwords2)]
  w = [word for word in w if (any(i.isdigit() for i in word) == False)]
  return w

def create_input(text, vocabulary, max_length=200):
  b           = get_words(text)
  input_row   = [vocabulary.index(word) for word in b if (word in vocabulary)][:max_length]
  pad_with    = [0]*(max_length-len(input_row))
  input_row   = pad_with + input_row
  return input_row


train, dev, test = delete_strange_authors(train, dev, test)
train, dev, test = choose_size(train, dev, test, fraction = 4/4)
vocabulary, authorsNames, idf = getVIA(train)

print('len of vocabulary  : ',len(vocabulary))
print('len of authorsNames: ',len(authorsNames))
print('len of idf         : ',len(idf))

def get_tf(words):
  tf = defaultdict(int)
  for word in words:
    tf[word] += 1
  tf = {x:(y/len(words)) for x,y in tf.items()}
  return tf

def unique(list1):
  unique_list = []
  for x in list1:
    if x not in unique_list:
      unique_list.append(x)
  return unique_list

def separate_dict_to_texts_and_author_lists(myset):
  texts = []
  authors = []
  for key, values in myset.items():
    for value in values:
      texts.append(value)
      authors.append(key)
  return texts, authors

texts, authors = separate_dict_to_texts_and_author_lists(train)
texts2, authors2 = separate_dict_to_texts_and_author_lists(dev)



#############################################
#                                           #
#             BACHERATOR FOR RNN            #
#                                           #
#############################################

def batch(indexes, authorsNames, vocabulary, idf, myset, target, n=1):
  max_length = 200
  for ndx in range(0, len(indexes), n):
    input = []
    output = []
    for i in indexes[ndx: ndx + n]:
      text, name  = myset[i], target[i]
      input_row   = create_input(text, vocabulary)
      input.append(input_row)
      output.append(authorsNames.index(name))
    yield input, output

# creates scores from models y,  noTexts x len(vocabulary)
def get_y_scores(pred):
  y_scores = []
  for items in pred:
    for item in list(items.cpu().data.numpy()):
      y_scores.append(list(item))
  return y_scores

# creates authors noTexts x len(authors) with 0 and 1 from list with authors indexes
def get_y_true(y_true_index_of_author):
  y_true = []
  for item in y_true_index_of_author:
    row = [0] * len(authorsNames)
    row[item] = 1
    y_true.append(row)
  return y_true

##### roc_auc_score takes np.arrays, the actual classes and the predicted classes in onehotencoding form



################################################################
###  this function evaluates the average score from batches  ###

def evaluate_score(y_true, y_scores):
  score = 0
  for tr, sc in zip(y_true, y_scores):
    score += roc_auc_score(tr, sc)
  score = score/len(y_true)
  return score

def load_model_from_checkpoint(resume_from):
  if os.path.isfile(resume_from):
    print("=> loading checkpoint '{}'".format(resume_from))
    checkpoint = torch.load(resume_from, map_location=lambda storage, loc: storage)
    pprint(checkpoint['state_dict'].keys())
    model.load_state_dict(checkpoint['state_dict'])

    print("=> loaded checkpoint '{}' (epoch {})".format(resume_from, checkpoint['epoch']))
  return checkpoint, model

def save_checkpoint(epoch, model, score, optimizer, filename='checkpoint.pth.tar'):
  '''
  :param state:       the state of the pytorch mode
  :param filename:    the name of the file in which we will store the model.
  :return:            Nothing. It just saves the model.
  '''
  state = {
      'epoch':            epoch,
      'state_dict':       model.state_dict(),
      'best_valid_score': score,
      'optimizer':        optimizer.state_dict(),
  }
  torch.save(state, filename)

  class Akis_model_RNN(torch.nn.Module):
      def __init__(self, vocabulary_size, emb_size, F, number_of_authors):
          super(Akis_model_RNN, self).__init__()
          self.embeds = nn.Embedding(vocabulary_size, emb_size)
          self.my_gru = nn.GRU(input_size=emb_size, hidden_size=F, batch_first=True, bidirectional=False)
          self.dropout = nn.Dropout(p=0.5)
          self.linear_layer_1 = nn.Linear(F, number_of_authors)
          self.loss = nn.CrossEntropyLoss()

      def forward(self, x, t):
          y = self.embeds(x)
          y = self.dropout(y)
          y, hn = self.my_gru(y)
          y, indices = torch.max(y, dim=1)
          y = self.linear_layer_1(y)
          l = self.loss(y, t)
          y = torch.softmax(y, dim=1)
          return y, l

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Akis_model_RNN(vocabulary_size=len(vocabulary), emb_size=100, F=len(authorsNames), number_of_authors=len(authorsNames)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam or SGD


# I AM TRAINING NOW
def train_one(model, rndTrain, authorsNames, vocabulary, idf, texts, authors, batch_size = 20):
  model.train()
  all_losses = []
  batcherator = batch(rndTrain, authorsNames, vocabulary, idf, texts, authors, batch_size)
  pbar = tqdm(batcherator, total=len(texts)/batch_size)
  for train_x, train_y in pbar:
    train_x = torch.LongTensor(train_x).to(device)
    train_y = torch.LongTensor(train_y).to(device)
    y, l = model(x=train_x, t=train_y)
    l.backward()   # backpropagation
    all_losses.append(l.cpu().item())
    optimizer.step()
    pbar.set_description('{}'.format(str(round(np.average(all_losses), 4))))
    optimizer.zero_grad()   # making the grad zero
  return model

# EVALUATION TIME

def eval_one(model, rndDev, authorsNames, vocabulary, idf, texts2, authors2, batch_size = 20):
  model.eval()
  batcherator = batch(rndDev, authorsNames, vocabulary, idf, texts2, authors2, batch_size)
  pred = []
  y_true = []
  y_predicted = []
  # print('Evaluation in progress....')
  for dev_x, dev_y in tqdm(batcherator, total=len(texts2)/batch_size):
    # dev_x = torch.FloatTensor(dev_x).to(device)
    dev_x = torch.LongTensor(dev_x).to(device)
    dev_y = torch.LongTensor(dev_y).to(device)
    y, l     = model(x=dev_x, t=dev_y)
    y_predicted.extend(y)
    ans = torch.argmax(y.cpu(), dim=1).data.numpy()
    pred.extend(ans)
    y_true.extend(dev_y.tolist())
  sc = accuracy_score( y_true,pred, normalize = True)
  return sc, y_true, y_predicted

##########################
###### PLOT lines  #######
def plot_line(train_score, dev_score):
  plt.rcParams["figure.figsize"] = (5,5)
  plt.rcParams["axes.grid"] = True
  matplotlib.rc('xtick', labelsize=15)
  matplotlib.rc('ytick', labelsize=15)
  fig, ax = plt.subplots()
  fig.suptitle('Train - Dev Lines', fontsize=16)
  plt.xlabel('epochs', fontsize=16)
  plt.ylabel('score', fontsize=16)
  ndx = [x+1 for x in range (len(train_score))]
  plt.plot( ndx , train_score, label="train")
  plt.plot( ndx , dev_score, label="dev")
  plt.legend(shadow=True, fontsize="large", loc="best")
  plt.ylim(0,1.1)   #  creates y scale from 0 to 1
  plt.show()


# We set seed value to be 1 for having the same shuffling every time
my_seed = 1
random.seed(my_seed)
torch.manual_seed(my_seed)
np.random.seed(my_seed)


epoch = 1
best_epoch = 1
best_score = float("nan")
train_score = []
dev_score = []
patience = 3
print('\n')
print('--- Start Training ---')
print('----------------------')
while patience != 0:
  rndTrain = list(range(0, len(texts)))
  random.shuffle(rndTrain)
  model = train_one(model, rndTrain, authorsNames, vocabulary, idf, texts, authors)
  score1, p1, p2 = eval_one(model, list(range(0, len(texts))), authorsNames, vocabulary, idf, texts, authors)
  score2, y_true, y_predicted = eval_one(model, list(range(0, len(texts2))), authorsNames, vocabulary, idf, texts2, authors2)
  train_score.append(score1)
  dev_score.append(score2)
  if math.isnan(best_score) or score2 > best_score:
    best_score = score2
    best_epoch = epoch
    patience = 3
    f = 'Texts160_E100_D05'
    save_checkpoint(epoch, model, epoch, optimizer, filename='/content/drive/My Drive/saved_files/paper/RNN_WITH_DROPOUT/{}.json'.format(f))
  else:
    patience -=1
  print('Train Score:  epoch: {} ---> score: {}'.format(epoch, score1))
  print('Dev Score  :  epoch: {} ---> score: {}'.format(epoch, score2))
  print('Best epoch: {} --- Best score: {}'.format(best_epoch, best_score))
  epoch += 1
  plot_line(train_score, dev_score)


test = dict(list(test.items())[0:50])
texts3, authors3 = separate_dict_to_texts_and_author_lists(test)

 # EVALUATION TIME



f = 'Texts160_E100_D05'
r1, r2 = load_model_from_checkpoint('/content/drive/My Drive/saved_files/paper/RNN_WITH_DROPOUT/{}.json'.format(f))
model = r2

model.eval()
rndTrain = list(range(0, len(texts3)))
random.shuffle(rndTrain)
batcherator = batch(rndTrain, authorsNames, vocabulary, idf, texts3, authors3, n = 20)
y_true = []  # list that collects the index of authors from texts
y_predicted = []   # list that collects the predicted values
b = 0
print('Evaluation in progress....')
for dev_x, dev_y in tqdm(batcherator, total=len(texts3)/20):
  dev_x = torch.LongTensor(dev_x).to(device)
  dev_y = torch.LongTensor(dev_y).to(device)
  y, l     = model(x=dev_x, t=dev_y)
  print(' batch ',b, ': ', l.cpu().data.numpy(), torch.argmax(y.cpu(), dim=1).data.numpy())
  ans = torch.argmax(y.cpu(), dim=1).data.numpy()
  y_predicted.extend(ans)
  y_true.extend(dev_y.tolist())
  b += 1


res = accuracy_score( y_true,y_predicted, normalize = True)
print('Test Score             :  ', res)
print('Test true results      :  ',y_true)
print('Test predicted results :  ',y_predicted)

def create_confusion_matrix_2d_array(true, predicted, noAuthors):
  table = [[0] * noAuthors] * noAuthors
  df = pd.DataFrame(table)
  for i, j in zip (true, predicted):
    df[int(j)][int(i)] += 1
  return df

cm = create_confusion_matrix_2d_array(y_true, y_predicted, 50)


# Confusion Matrix PLOT

sns.set(font_scale=2)
array = np.array(cm)
df = pd.DataFrame(array, index = [i for i in authorsNames], columns = [i for i in authorsNames])
fig, ax = plt.subplots()
plt.figure(figsize = (25,25))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.title('Confusion Matrix')
sns.heatmap(df, annot = True)

plt.savefig('/content/drive/My Drive/saved_files/paper/images/RNN_WITH_DROPOUT_{}.png'.format(f))