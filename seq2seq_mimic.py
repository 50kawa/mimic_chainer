# -*- coding:utf-8 -*-
"""
Sample script of Sequence to Sequence model.
You can also use Batch and GPU.
This model is based on below paper.

Ilya Sutskever, Oriol Vinyals, and Quoc V. Le.
Sequence to sequence learning with neural networks.
In Advances in Neural Information Processing Systems (NIPS 2014).
"""
import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import cupy as cp
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable

# global variable (initialize)
xp = np


class Encoder(chainer.Chain):
    def __init__(self, vocab_size, character_embed_size,embed_size, hidden_size, batch_size,use_dropout):
        super(Encoder, self).__init__(
            word_embed=L.EmbedID(vocab_size, character_embed_size,ignore_label=-1),
            bi_lstm=L.NStepBiLSTM(n_layers=1, in_size=character_embed_size,out_size=hidden_size, dropout=use_dropout),
            h_e=L.Linear(hidden_size*2, hidden_size*2),
            e_o=L.Linear(hidden_size * 2, embed_size),
        )
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.use_dropout=use_dropout

    def __call__(self, x_list):
        hx = None
        cx = None
        xs_f = []
        x_list=[xp.array(x, dtype=xp.int32) for x in x_list]
        for i, x in enumerate(x_list):
            x = self.word_embed(Variable(x))  # Word IndexからWord Embeddingに変換
            x = F.dropout(x, ratio=self.use_dropout)
            xs_f.append(x)
        hy, _, _ = self.bi_lstm(hx=hx, cx=cx, xs=xs_f)
        return self.e_o(F.tanh(self.h_e(F.concat(hy,axis=1))))


class Mimic(chainer.Chain):

    def __init__(self, vocab_size, character_embed_num,feature_num, hidden_num, batch_size, gpu_flg,use_dropout):
        """
        :param vocab_size: input vocab size
        :param feature_num: size of feature layer (embed layer)
        :param hidden_num: size of hidden layer
        :return:
        """
        global xp
        xp = cuda.cupy if gpu_flg >= 0 else np

        self.vocab_size = vocab_size
        self.hidden_num = hidden_num
        self.batch_size = batch_size

        super(Mimic, self).__init__(
            enc=Encoder(vocab_size, character_embed_num,feature_num, hidden_num, batch_size,use_dropout)     # encoder
        )

    def encode(self, input_batch,output_batch,train):
        """
        Input batch of sequence and update self.c (context vector) and self.h (hidden vector)
        :param input_batch: batch of input text embed id ex.) [[ 1, 0 ,14 ,5 ], [ ...] , ...]
        :param train : True or False
        """
        input_emb=self.enc(input_batch)
        if train:
            t = xp.array(output_batch, dtype=xp.float32)
            t = chainer.Variable(t)
            return F.mean_squared_error(input_emb, t)
        else:
            return input_emb
