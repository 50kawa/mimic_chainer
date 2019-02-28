# -*- coding:utf-8 -*-
"""
"""
import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import cupy as cp
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda,  serializers, Variable
import sys
sys.path.append('C:/Users/isogawa/word2vec_en/mimic/subword-nmt/subword_nmt')
from apply_bpe import BPE
from util import Dataread

# global variable (initialize)
xp = np


def load_model(conf, initial_embedding=None):
    corpus = Dataread()
    corpus.load(conf["datapack_dir"], conf["compareword"])

    model = Model(len(corpus.char_dict), corpus.max_input_len, conf, initial_embedding=initial_embedding)

    return model, corpus


class SplitWord():
    def __init__(self, config):
        if "BPE" in config:
            if "BPE" in config["BPE"]:
                self.way = config["BPE"]
                if config["BPE"] == "BPE":
                    self.bpe = BPE(codecs.open('D:/wiki_20180801/bpe.code', encoding='utf-8'), separator='')
                elif config["BPE"] == "BPE1000":
                    self.bpe = BPE(codecs.open('D:/wiki_20180801/bpe1000.code', encoding='utf-8'), separator='')
                else:
                    print("BPE define error")
                    exit()
            else:
                self.way = config["BPE"]
        else:
            self.way = "Normal"

    def __call__(self, word):
        if self.way == "BPE":
            return self.bpe.process_line(word).split(" ")
        elif self.way == "Ngram":
            list_of_ngram = []
            for i in range(3, 7):
                list_of_ngram.extend(ngram(word, i))
            return list_of_ngram
        else:
            return word


class Interpreter():
    def __init__(self, config):
        self.split_word = SplitWord(config)
        self.config = config
        self.config["GPU"] = 0
        self.config["batchsize"] = 1
        self.model, self.corpus = load_model(self.config)
        if self.config["GPU"] == 0:
            self.model.to_gpu()
        if "load_train" in self.config and self.config["load_train"]:
            serializers.load_hdf5('C:/Users/isogawa/word2vec_en/mimic/data/'+config["modelname"]+'_best_train_loss.model', self.model)
        else:
            serializers.load_hdf5('C:/Users/isogawa/word2vec_en/mimic/data/'+config["modelname"]+'_best.model', self.model)

    def __call__(self, word):
        # run conversation system
        input = []
        if "Omozi" in self.config and self.config["Omozi"] == "Use":
            wordlow = word
        else:
            wordlow = word.lower()
        for w in self.split_word(wordlow):
            if w not in self.corpus.char_dict:
                input.append(-1)
            else:
                input.append(self.corpus.char_dict[w])
        input = [input]
        return cuda.to_cpu(self.model.make_vector(input).data[0])


class Encoder(chainer.Chain):
    def __init__(self, vocab_size, character_embed_size, embed_size, hidden_size,
                 batch_size, use_dropout, initial_embedding):
        super(Encoder, self).__init__(
            word_embed=L.EmbedID(vocab_size, character_embed_size, initialW=initial_embedding, ignore_label=-1),
            bi_lstm=L.NStepBiLSTM(n_layers=1, in_size=character_embed_size, out_size=hidden_size, dropout=use_dropout),
            h_e=L.Linear(hidden_size*2, hidden_size*2),
            e_o=L.Linear(hidden_size * 2, embed_size),
        )
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.use_dropout = use_dropout

    def __call__(self, x_list):
        hx = None
        cx = None
        xs_f = []
        for x in x_list:
            x = self.word_embed(Variable(x))  # Word IndexからWord Embeddingに変換
            x = F.dropout(x, ratio=self.use_dropout)
            xs_f.append(x)
        hy, _, _ = self.bi_lstm(hx=hx, cx=cx, xs=xs_f)
        return self.e_o(F.tanh(self.h_e(F.concat(hy, axis=1))))


class EncoderGRU(chainer.Chain):
    def __init__(self, vocab_size, character_embed_size, embed_size, hidden_size,
                 batch_size, use_dropout, initial_embedding):
        super(EncoderGRU, self).__init__(
            word_embed=L.EmbedID(vocab_size, character_embed_size, initialW=initial_embedding, ignore_label=-1),
            bi_gru=L.NStepBiGRU(n_layers=1, in_size=character_embed_size, out_size=hidden_size, dropout=use_dropout),
            h_e=L.Linear(hidden_size*2, hidden_size*2),
            e_o=L.Linear(hidden_size * 2, embed_size),
        )
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.use_dropout = use_dropout

    def __call__(self, x_list):
        hx = None
        xs_f = []
        for x in x_list:
            x = self.word_embed(Variable(x))  # Word IndexからWord Embeddingに変換
            x = F.dropout(x, ratio=self.use_dropout)
            xs_f.append(x)
        hy, _ = self.bi_gru(hx=hx, xs=xs_f)
        return self.e_o(F.tanh(self.h_e(F.concat(hy, axis=1))))


class EncoderSum(chainer.Chain):
    def __init__(self, vocab_size, character_embed_size, embed_size, hidden_size,
                 batch_size, use_dropout, initial_embedding):
        super(EncoderSum, self).__init__(
            word_embed=L.EmbedID(vocab_size, character_embed_size, initialW=initial_embedding, ignore_label=-1),
        )
        self.hidden_size = hidden_size
        self.batch_size = batch_size

    def __call__(self, x_list):
        return F.sum(self.word_embed(xp.array(x_list, dtype=xp.int32)), axis=1)


class EncoderSumFF(chainer.Chain):
    def __init__(self, vocab_size, character_embed_size, embed_size, hidden_size,
                 batch_size, use_dropout, initial_embedding):
        super(EncoderSumFF, self).__init__(
            word_embed=L.EmbedID(vocab_size, character_embed_size, initialW=initial_embedding, ignore_label=-1),
            h_e=L.Linear(embed_size, embed_size),
            e_o=L.Linear(embed_size, embed_size),

        )
        self.hidden_size = hidden_size
        self.batch_size = batch_size

    def __call__(self, x_list):
        return self.e_o(F.tanh(self.h_e(F.sum(self.word_embed(xp.array(x_list, dtype=xp.int32)), axis=1))))


class EncoderCNN(chainer.Chain):
    def __init__(self, vocab_size, embed_size, batch_size, max_len):
        super(EncoderCNN, self).__init__(
            conv_1=L.Convolution2D(1, 200, ksize=(1, vocab_size)),
            conv_2=L.Convolution2D(1, 200, ksize=(2, vocab_size)),
            conv_3=L.Convolution2D(1, 200, ksize=(3, vocab_size)),
            conv_4=L.Convolution2D(1, 200, ksize=(4, vocab_size)),
            conv_5=L.Convolution2D(1, 250, ksize=(5, vocab_size)),
            conv_6=L.Convolution2D(1, 300, ksize=(6, vocab_size)),
            conv_7=L.Convolution2D(1, 350, ksize=(7, vocab_size)),
            highway_1=L.Highway(1700, activate=F.tanh),
            highway_2=L.Highway(1700, activate=F.tanh),
            linear=L.Linear(1700, embed_size)
        )
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.max_len = max_len
        self.identity = xp.identity(vocab_size)

    def __call__(self, x_list):
        xs_f = F.embed_id(xp.array(x_list, dtype=xp.int32), self.identity, ignore_label=-1)
        xs_f = xp.reshape(xs_f, (self.batch_size, 1, self.max_len, self.vocab_size))
        conv1 = self.conv_1(xs_f)  # (batch, max(200, width*50), len(word))
        pooled1 = F.sum(F.max_pooling_2d(F.tanh(conv1), 3, 3), axis=2)  # pool->(batch, max(200, width*50), len(word)/3)
        conv2 = self.conv_2(F.pad(xs_f, [(0, 0), (0, 0), (1, 0), (0, 0)], 'constant'))
        pooled2 = F.sum(F.max_pooling_2d(F.tanh(conv2), 3, 3), axis=2)
        conv3 = self.conv_3(F.pad(xs_f, [(0, 0), (0, 0), (1, 1), (0, 0)], 'constant'))
        pooled3 = F.sum(F.max_pooling_2d(F.tanh(conv3), 3, 3), axis=2)
        conv4 = self.conv_4(F.pad(xs_f, [(0, 0), (0, 0), (2, 1), (0, 0)], 'constant'))
        pooled4 = F.sum(F.max_pooling_2d(F.tanh(conv4), 3, 3), axis=2)
        conv5 = self.conv_5(F.pad(xs_f, [(0, 0), (0, 0), (2, 2), (0, 0)], 'constant'))
        pooled5 = F.sum(F.max_pooling_2d(F.tanh(conv5), 3, 3), axis=2)
        conv6 = self.conv_6(F.pad(xs_f, [(0, 0), (0, 0), (3, 2), (0, 0)], 'constant'))
        pooled6 = F.sum(F.max_pooling_2d(F.tanh(conv6), 3, 3), axis=2)
        conv7 = self.conv_7(F.pad(xs_f, [(0, 0), (0, 0), (3, 3), (0, 0)], 'constant'))
        pooled7 = F.sum(F.max_pooling_2d(F.tanh(conv7), 3, 3), axis=2)

        e = F.concat((pooled1, pooled2, pooled3, pooled4, pooled5, pooled6, pooled7), axis=1) # (batch, max(200, width*50)*7)
        return self.linear(self.highway_2(self.highway_1(xp.reshape(e, (self.batch_size, 1700)))))


class Model(chainer.Chain):

    def __init__(self, vocab_size, max_input_len, conf, initial_embedding=None):
        """
        :param vocab_size: input vocab size
        :param feature_num: size of feature layer (embed layer)
        :param hidden_num: size of hidden layer
        :return:
        """
        global xp
        xp = cuda.cupy if conf["GPU"] >= 0 else np

        self.vocab_size = vocab_size
        self.hidden_num = conf["hidden_num"]
        self.batch_size = conf["batchsize"]

        self.config = conf
        self.max_input_len = max_input_len
        if conf["NN_model"] == "RNN":
            super(Model, self).__init__(
                enc=Encoder(vocab_size, conf["character_embed_num"], conf["feature_num"], conf["hidden_num"],
                            conf["batchsize"], conf["dropout"], initial_embedding)
            )
        elif conf["NN_model"] == "GRU":
            super(Model, self).__init__(
                enc=EncoderGRU(vocab_size, conf["character_embed_num"], conf["feature_num"], conf["hidden_num"],
                               conf["batchsize"], conf["dropout"], initial_embedding)
            )
        elif conf["NN_model"] == "SUM":
            super(Model, self).__init__(
                enc=EncoderSum(vocab_size, conf["character_embed_num"], conf["feature_num"], conf["hidden_num"],
                               conf["batchsize"], conf["dropout"], initial_embedding)
            )
        elif conf["NN_model"] == "CNN":
            super(Model, self).__init__(
                enc=EncoderCNN(vocab_size, conf["feature_num"], conf["batchsize"], max_input_len)
            )
        elif conf["NN_model"] == "SUMFF":
            super(Model, self).__init__(
                enc=EncoderSumFF(vocab_size, conf["character_embed_num"], conf["feature_num"], conf["hidden_num"],
                                 conf["batchsize"], conf["dropout"], initial_embedding)
            )
        else:
            print("NN_model not defined")
            exit()

    def train(self, input_batch, input_wv_batch, words_batch=None, output_batch=None):
        """
        Input batch of sequence and update self.c (context vector) and self.h (hidden vector)
        :param input_batch: batch of input text embed id ex.) [[ 1, 0 ,14 ,5 ], [ ...] , ...]
        :param train : True or False
        """
        input_emb = self.enc(input_batch)
        if self.config["use_relation"] == "Similarity":
            compare_word_embedding = chainer.Variable(xp.array(words_batch, dtype=xp.float32))
            # cos類似度の計算
            a = F.tile(F.expand_dims(F.normalize(input_emb), 1), (1, self.config["compareword"], 1))  # (batch, cw, dim)
            b = F.normalize(compare_word_embedding, axis=2)  # (batch, cw, dim)
            cos_sim = F.sum(a * b, axis=2)
            # 誤差の計算
            t_v = xp.array(input_wv_batch, dtype=xp.float32)
            t = xp.array(output_batch, dtype=xp.float32)
            t = chainer.Variable(t)
            if self.config["Objective_Function"] == "SE":
                return (F.mean_squared_error(input_emb, t_v) * self.config["feature_num"] + F.mean_squared_error(cos_sim, t) * self.config["feature_num"]) / 2
            else:
                return (F.mean_squared_error(input_emb, t_v) + F.mean_squared_error(cos_sim, t)) / 2

        else:
            t = xp.array(input_wv_batch, dtype=xp.float32)
            t = chainer.Variable(t)
            if self.config["Objective_Function"] == "SE":
                return F.mean_squared_error(input_emb, t) * self.config["feature_num"]
            else:
                return F.mean_squared_error(input_emb, t)

    def make_vector(self, input_batch):
        """
        Input batch of sequence and update self.c (context vector) and self.h (hidden vector)
        :param input_batch: batch of input text embed id ex.) [[ 1, 0 ,14 ,5 ], [ ...] , ...]
        :param train : True or False
        """
        if self.config["NN_model"] == "CNN":
            input_batch = [[input_batch[0][i] if i < len(input_batch[0]) else -1 for i in range(self.max_input_len)]]
        return self.enc(xp.array(input_batch, dtype=xp.int32))
