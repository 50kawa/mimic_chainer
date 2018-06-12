# -*- coding:utf-8 -*-

import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import argparse
import unicodedata
import pickle
import numpy as np
import matplotlib.pyplot as plt
from nltk import word_tokenize
from chainer import serializers, cuda
from util import Mimicdata
from seq2seq_mimic import Mimic


# path info
DATA_DIR = './data/corpus/'
MODEL_PATH = './data/w2v_54_most.model'
TRAIN_LOSS_PATH = './data/loss_train_data.pkl'
TEST_LOSS_PATH = './data/loss_test_data.pkl'
BLEU_SCORE_PATH = './data/bleu_score_data.pkl'
WER_SCORE_PATH = './data/wer_score_data.pkl'

# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', default='data/pair_corpus.txt', type=str, help='Data file directory')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=300, type=int, help='number of epochs to learn')
parser.add_argument('--character_embed_num', '-c', default=20, type=int, help='dimension of feature layer')
parser.add_argument('--feature_num', '-f', default=300, type=int, help='dimension of feature layer')
parser.add_argument('--hidden_num', '-hi', default=600, type=int, help='dimension of hidden layer')
parser.add_argument('--batchsize', '-b', default=1000, type=int, help='learning minibatch size')
parser.add_argument('--testsize', '-t', default=1000, type=int, help='number of text for testing a model')
parser.add_argument('--vocabsize', '-v', default=26, type=int, help='number of vocab_size')
parser.add_argument('--maxtextlength', '-m', default=25, type=int, help='number of maxtextlength')
parser.add_argument('--lang', '-l', default='ja', type=str, help='the choice of a language (Japanese "ja" or English "en" )')

args = parser.parse_args()

data_file = args.data
n_epoch = args.epoch
character_embed_num=args.character_embed_num
feature_num = args.feature_num
hidden_num = args.hidden_num
batchsize = args.batchsize
testsize = args.testsize
vocabsize=args.vocabsize
max_sequence_len=args.maxtextlength

# GPU settings
gpu_device = args.gpu
if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(gpu_device).use()


def parse_ja_text(text):
    """
    Function to parse Japanese text.
    :param text: string: sentence written by Japanese
    :return: list: parsed text
    """
    import MeCab
    mecab = MeCab.Tagger("mecabrc")
    mecab.parse('')

    # list up noun
    mecab_result = mecab.parseToNode(text)
    parse_list = []
    while mecab_result is not None:
        if mecab_result.surface != "":  # ヘッダとフッタを除外
            parse_list.append(unicodedata.normalize('NFKC', mecab_result.surface).lower())
        mecab_result = mecab_result.next

    return parse_list


def interpreter(model_path):
    """
    Run this function, if you want to talk to seq2seq model.
    if you type "exit", finish to talk.
    :param data_path: the path of corpus you made model learn
    :param model_path: the path of model you made learn
    :return:
    """
    # call dictionary class
    corpus=Mimicdata()
    corpus.load_interpreter("datapack_w2v.pkl")

    # rebuild seq2seq model
    model = Mimic(vocabsize, character_embed_num,feature_num=feature_num,hidden_num=hidden_num, batch_size=batchsize, gpu_flg=args.gpu,use_dropout=0)
    serializers.load_hdf5(model_path, model)
    vector_dict={}
    with open("sim_wordset.pkl","rb")as f:
        word_list=pickle.load(f)

    # run conversation system
    for word in word_list:
        input=[]
        wordlow=word.lower()
        for w in wordlow:
            if w not in corpus.char_dict:
                input.append(-1)
            else:
                input.append(corpus.char_dict[w])
        input=[input]
        vector = model.encode(input, 0, train=False)
        vector_dict[word]=vector.data[0]

    with open("mimic_w2v_vectordict.pkl","wb")as f:
        pickle.dump(vector_dict,f)



if __name__ == '__main__':
    interpreter(MODEL_PATH)
