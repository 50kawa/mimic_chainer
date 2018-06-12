# -*- coding:utf-8 -*-

import re
import pickle
import unicodedata
import numpy as np
from gensim import corpora
from nltk import word_tokenize


class Mimicdata:
    """
    Dictionary Class for Japanese
    """
    def __init__(self):
        """
       if file_path is not None:
           self._construct_dict(file_path, batch_size, size_filter)
       """

    def load(self, load_dir):
        with open(load_dir , "rb") as f:
            datapack = pickle.load(f)
        self.data=list(zip(datapack[0],datapack[1]))

    def load_interpreter(self, load_dir):
        with open(load_dir , "rb") as f:
            datapack = pickle.load(f)
        self.char_dict=datapack[2]
        self.char_id_dict=datapack[3]


