# -*- coding:utf-8 -*-

import re
import pickle
import unicodedata
import numpy as np
from gensim.models import KeyedVectors
from nltk import word_tokenize


class Dataread:
    def __init__(self):
        """
       """

    def load(self, load_dir, cw_num):
        with open(load_dir, "rb") as f:
            datapack = pickle.load(f)
        cw_num = cw_num//2

        # make train data
        train_data = datapack["train_data"]
        cw_top = [a[:cw_num] for a in train_data["compare_words_pos"]]
        sim_top = [a[:cw_num] for a in train_data["simlists_pos"]]
        self.train_data = [train_data["input"], train_data["input_word"], cw_top, sim_top]

        # make dev data
        if "dev_data" in datapack:
            dev_data = datapack["dev_data"]
            cw_top = [a[:cw_num] for a in dev_data["compare_words_pos"]]
            sim_top = [a[:cw_num] for a in dev_data["simlists_pos"]]
            self.dev_data = [dev_data["input"], dev_data["input_word"], cw_top, sim_top]
            self.max_input_len = max(len(x) for x in train_data["input"] + dev_data["input"])
        else:
            self.max_input_len = max(len(x) for x in train_data["input"])

        # make test data
        if "test_data" in datapack:
            test_data = datapack["test_data"]
            self.test_data = test_data["input_word"]

        self.char_dict = datapack["char_dict"]
        self.char_id_dict = datapack["char_id_dict"]
        if "random_cw_list" in datapack:
            self.random_cw_list = datapack["random_cw_list"]
        elif "random_cw_set" in datapack:
            self.random_cw_list = list(datapack["random_cw_set"])
        else:
            print("cw list is not found")



    def load_interpreter(self, load_dir):
        with open(load_dir, "rb") as f:
            datapack = pickle.load(f)
        self.char_dict = datapack["char_dict"]
        self.char_id_dict = datapack["char_id_dict"]


class PWIMdata:
    def __init__(self):
        """
       """

    def load(self, config):
        from data_make_PWIM import datamake
        wordvector_dict = KeyedVectors.load_word2vec_format(config["word_vector_dir"], binary=False)

        self.train_data, self.char_dict, self.char_id_dict, self.random_cw_list = datamake(config, wordvector_dict)
        self.wordvector_model = wordvector_dict
        self.max_input_len = max(len(x) for x in self.train_data[0])
        with open(config["modelname"] + "_chardict.pkl", "wb") as f:
            pickle.dump({"char_dict": self.char_dict, "char_id_dict": self.char_id_dict}, f)


    def load_interpreter(self, config):
        with open(config["modelname"] + "_chardict.pkl", "rb") as f:
            datapack = pickle.load(f)
        self.char_dict = datapack["char_dict"]
        self.char_id_dict = datapack["char_id_dict"]

