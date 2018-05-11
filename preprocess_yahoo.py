# -*- coding: utf-8 -*-
"""
Created on Mon May  1 01:53:46 2017

@author: DinghanShen
"""

import csv
import numpy as np
import os
import re
import cPickle
import string
import pdb

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
#==============================================================================
loadpath = './test.csv'

x = []
with open(loadpath, 'rb') as f:
    for line in f:
        x.append(line)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),\.!?]", " ", string)
    #string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"e\.g\.,", " ", string)
    string = re.sub(r"a\.k\.a\.", " ", string)
    string = re.sub(r"i\.e\.,", " ", string)
    string = re.sub(r"i\.e\.", " ", string)
    #string = re.sub(r"\'ve", " \'ve", string)
    #string = re.sub(r"\'", "", string)
    #string = re.sub(r"\'re", " \'re", string)
    #string = re.sub(r"\'d", " \'d", string)
    #string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"br", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"u\.s\.", " us ", string)
    return string.strip().lower()


lab = []
sent = []
vocab = {}

for i in range(60000):
    # lab.append(clean_str(x[i].split(",")[0]))
    m = re.search(',', x[i])
    rest = x[i][m.start()+1:]
    m = re.search(',', rest)
    temp = clean_str(rest[:m.start()]).split()
    # temp = clean_str(x[i][m.start()+1 : n + 1]).split()
    temp = [ j if not is_number(j) else '0' for j in temp]
    if len(temp) > 300:
        lab.append(clean_str(x[i].split(",")[0]))
        temp = temp[:300]
        sent.append(temp)
    elif len(temp) <= 5: # remove too short question
        continue
    else:
        lab.append(clean_str(x[i].split(",")[0]))
        sent.append(temp)
    t = set(temp)
    for word in t:
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1

loadpath = './train.csv'

x = []
with open(loadpath, 'rb') as f:
    for line in f:
        x.append(line)

train_lab = []
train_sent = []

for i in range(len(x)):
    # train_lab.append(clean_str(x[i].split(",")[0]))
    m = re.search(',', x[i])
    rest = x[i][m.start()+1:]
    m = re.search(',', rest)
    temp = clean_str(rest[:m.start()]).split()

    # temp = clean_str(x[i][m.start()+1: n+ 1]).split()
    temp = [ j if not is_number(j) else '0' for j in temp]
    if len(temp) > 300:
        train_lab.append(clean_str(x[i].split(",")[0]))
        temp = temp[:300]
        train_sent.append(temp)
    elif len(temp) <= 5: # remove too short question
        continue
    else:
        train_lab.append(clean_str(x[i].split(",")[0]))
        train_sent.append(temp)
    t = set(temp)
    for word in t:
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1
#==============================================================================

print('create ixtoword and wordtoix lists...')

v = [x for x, y in vocab.iteritems() if y >= 30]

# create ixtoword and wordtoix lists
ixtoword = {}
# period at the end of the sentence. make first dimension be end token
ixtoword[0] = 'END'
ixtoword[1] = 'UNK'
wordtoix = {}
wordtoix['END'] = 0
wordtoix['UNK'] = 1
ix = 2
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

def convert_word_to_ix(data):
    result = []
    for sent in data:
        temp = []
        for w in sent:
            if w in wordtoix:
                temp.append(wordtoix[w])
            else:
                temp.append(1)
        temp.append(0)
        result.append(temp)
    return result

train_x = train_sent[:1100000]
train_y = train_lab[:1100000]
val_x = train_sent[1100000:]
val_y = train_lab[1100000:]
test_x = sent
test_y = lab


train_text = [' '.join(s) for s in train_x]
val_text = [' '.join(s) for s in val_x]
test_text = [' '.join(s) for s in test_x]



train_x = convert_word_to_ix(train_x)
val_x = convert_word_to_ix(val_x)
test_x = convert_word_to_ix(test_x)




cPickle.dump([train_x, val_x, test_x, train_text, val_text, test_text, train_y, val_y, test_y, wordtoix, ixtoword], open("yahoo4char.p", "wb"))

pdb.set_trace()
