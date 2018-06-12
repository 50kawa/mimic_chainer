#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import codecs
import pickle
import re
from gensim.models import KeyedVectors

repatter = re.compile(r"^[a-z]+$")

modelw=KeyedVectors.load_word2vec_format('enwiki.bin', binary=True,)
wordlist=model.vocab.keys()

input=[]
output=[]
batch=0
for word in wordlist:
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
        if batch>999999:
            break

datapack=[input,output,char_dict,char_id_dict]
with open("datapack_hundredthousand.pkl","wb") as f:
    pickle.dump(datapack,f)

print(len(char_dict))
