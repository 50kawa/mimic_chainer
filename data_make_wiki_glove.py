#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import codecs
import pickle
import re
from gensim.models import KeyedVectors

repatter = re.compile(r"^[a-z]+$")

char_dict={}
char_id_dict={}
model= KeyedVectors.load_word2vec_format('D:/glove.840B.300d/glove.840B.300d.read.txt', binary=False,unicode_errors="ignore")
with open("../wordlist_orderedbyfreq.pkl","rb")as f:
    wordlist=pickle.load(f) #len2176135?

input=[]
output=[]
batch=0
for word in wordlist:
    if word not in model:
        continue
    word_row=word.lower()
    if repatter.match(word_row):
        id_list=[]

        for c in word_row:
            if c not in char_dict:
                id=len(char_dict)
                char_dict[c]=id
                char_id_dict[id]=c
            id_list.append(char_dict[c])
        input.append(tuple(id_list))
        output.append(model[word])
        batch+=1
        if batch>99999:
            break

datapack=[input,output,char_dict,char_id_dict]
with open("datapack_GloVe.pkl","wb") as f:
    pickle.dump(datapack,f)

print(len(char_dict))
