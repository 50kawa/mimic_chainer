#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import codecs
import pickle
import codecs
import re
import argparse
import yaml
import random
from tqdm import tqdm
import numpy as np
from model import SplitWord


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def ngram(word, n):
    return list(zip(*(word[i:] for i in range(n))))


def zijougosa(v1, v2):
    return np.sum((v1-v2)**2)



if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--configfile', '-c', default="", type=str, help='')
    args = parser.parse_args()
    with open(args.configfile, "r+") as f:
        config = yaml.load(f)

    testsize = config["testsize"]
    devsize = config["devsize"]
    trainsize = config["trainsize"]
    num_of_choise = config["compareword"]  # 今はランダムにnこ単語を選んで類似度の比較に使う。類似している語とそれ以外を半々に

    repatter = re.compile(r"^[a-z]+$")
    split_word = SplitWord(config)
    if config["wordvector"] == "FastText":
        from gensim.models.wrappers import FastText
        model = FastText.load_fasttext_format('fasttext.sgns.bin')
        with open('word_lists.pkl', 'rb') as f:
            word_lists = pickle.load(f)
    elif config["wordvector"] == "FastText_Omozi":
        from gensim.models.wrappers import FastText
        model = FastText.load_fasttext_format('fasttextOmozi.bin')
        with open('word_lists_Omozi.pkl', 'rb') as f:
            word_lists = pickle.load(f)
    elif config["wordvector"] == "FastText_Kyodo":
        from gensim.models.wrappers import FastText
        model = FastText.load_fasttext_format('fasttextKyodo.bin')
        with open('word_lists_Kyodo.pkl', 'rb') as f:
            word_lists = pickle.load(f)
    else:
        print("wordvector not found")
        exit()

    char_dict = {}
    char_id_dict = {}
    random_cw_set = set(model.wv.vocab)

    if "overfit" in config and config["overfit"]:
        ##########################
        #     make train_data    #
        ##########################
        train_data = {"input": [], "input_word": [], "compare_words_pos": [], "simlists_pos": []}
        batch = 0
        for word in tqdm(word_lists["train_data"] + word_lists["dev_data"] + word_lists["test_data"]):
            word_row = word.lower()
            # char_indexに直してinputを決定
            id_list = []
            for c in split_word(word_row):
                if c not in char_dict:
                    id = len(char_dict)
                    char_dict[c] = id
                    char_id_dict[id] = c
                id_list.append(char_dict[c])
            train_data["input"].append(tuple(id_list))

            # 元の単語を保存
            train_data["input_word"].append(word)

            # 比較する語の抽出、平均二乗誤差が小さい順に取る,simlistsに予め二乗誤差を計算して保存しておく
            # 抽出した語はランダムに選ぶ語のリストから外す
            l = model.most_similar(positive=[word], topn=num_of_choise // 2)
            cwlist_top = [a[0] for a in l]
            simlist_top = []
            for c_w in cwlist_top:
                simlist_top.append(cos_sim(model[word], model[c_w]))
            random_cw_set -= set(cwlist_top)
            train_data["compare_words_pos"].append(cwlist_top)
            train_data["simlists_pos"].append(simlist_top)

        datapack = {"train_data": train_data,
                    "char_dict": char_dict,
                    "char_id_dict": char_id_dict,
                    "random_cw_list": list(random_cw_set)
                    }
        with open(config["datapack_dir"], "wb") as f:
            pickle.dump(datapack, f)

        print(len(char_dict))

    else:
        ##########################
        #     make train_data    #
        ##########################
        train_data = {"input": [], "input_word": [], "compare_words_pos": [], "simlists_pos": []}
        batch = 0
        for word in tqdm(word_lists["train_data"]):
            word_row = word.lower()
            # char_indexに直してinputを決定
            id_list = []
            for c in split_word(word_row):
                if c not in char_dict:
                    id = len(char_dict)
                    char_dict[c] = id
                    char_id_dict[id] = c
                id_list.append(char_dict[c])
            train_data["input"].append(tuple(id_list))

            # 元の単語を保存
            train_data["input_word"].append(word)

            # 比較する語の抽出、平均二乗誤差が小さい順に取る,simlistsに予め二乗誤差を計算して保存しておく
            # 抽出した語はランダムに選ぶ語のリストから外す
            l = model.most_similar(positive=[word], topn=num_of_choise//2)
            cwlist_top = [a[0] for a in l]
            simlist_top = []
            for c_w in cwlist_top:
                simlist_top.append(cos_sim(model[word], model[c_w]))
            random_cw_set -= set(cwlist_top)
            train_data["compare_words_pos"].append(cwlist_top)
            train_data["simlists_pos"].append(simlist_top)


        ##########################
        #     make dev_data      #
        ##########################
        dev_data = {"input": [], "input_word": [], "compare_words_pos": [], "simlists_pos": []}
        batch = 0
        for word in word_lists["dev_data"]:
            word_row = word.lower()
            # char_indexに直してinputを決定
            id_list = []
            for c in split_word(word_row):
                if c not in char_dict:
                    id_list.append(-1)
                else:
                    id_list.append(char_dict[c])
            dev_data["input"].append(tuple(id_list))

            # 元の単語を保存
            dev_data["input_word"].append(word)

            # 比較する語の抽出、平均二乗誤差が小さい順に取る,simlistsに予め二乗誤差を計算して保存しておく
            # 抽出した語はランダムに選ぶ語のリストから外す
            l = model.most_similar(positive=[word], topn=num_of_choise//2)
            cwlist_top = [a[0] for a in l]
            simlist_top = []
            for c_w in cwlist_top:
                simlist_top.append(cos_sim(model[word], model[c_w]))
            random_cw_set -= set(cwlist_top)
            dev_data["compare_words_pos"].append(cwlist_top)
            dev_data["simlists_pos"].append(simlist_top)


        ##########################
        #     make test_data     #
        ##########################
        test_data = {"input": [], "input_word": []}
        batch = 0
        for word in word_lists["test_data"]:
            word_row = word.lower()
            # char_indexに直してinputを決定
            id_list = []
            for c in split_word(word_row):
                if c not in char_dict:
                    id_list.append(-1)
                else:
                    id_list.append(char_dict[c])
            test_data["input"].append(tuple(id_list))

            # 元の単語を保存
            test_data["input_word"].append(word)

        datapack = {"test_data": test_data,
                    "dev_data": dev_data,
                    "train_data": train_data,
                    "char_dict": char_dict,
                    "char_id_dict": char_id_dict,
                    "random_cw_list": list(random_cw_set)
                    }
        with open(config["datapack_dir"], "wb") as f:
            pickle.dump(datapack, f)

        print(len(char_dict))
