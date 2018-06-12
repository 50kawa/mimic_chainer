# -*- coding:utf-8 -*-
"""
Sample script of Sequence to Sequence model for ChatBot.
This is a train script for seq2seq.py
You can also use Batch and GPU.
args: --gpu (flg of GPU, if you want to use GPU, please write "--gpu 1")
"""

import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import pickle
import argparse
import nltk
import numpy as np
import random
import chainer
from chainer import cuda, optimizers, serializers
from util import Mimicdata
from seq2seq_mimic import Mimic


# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', default='data/pair_corpus.txt', type=str, help='Data file directory')
parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=60, type=int, help='number of epochs to learn')
parser.add_argument('--character_embed_num', '-c', default=20, type=int, help='dimension of feature layer')
parser.add_argument('--feature_num', '-f', default=300, type=int, help='dimension of feature layer')
parser.add_argument('--hidden_num', '-hi', default=600, type=int, help='dimension of hidden layer')
parser.add_argument('--batchsize', '-b', default=1000, type=int, help='learning minibatch size')
parser.add_argument('--testsize', '-t', default=1000, type=int, help='number of text for testing a model')
parser.add_argument('--vocabsize', '-v', default=26, type=int, help='number of vocab_size')
parser.add_argument('--training_rate', '-tr', default=0.01, type=float, help='training rate')
parser.add_argument('--lang', '-l', default='ja', type=str, help='the choice of a language (Japanese "ja" or English "en" )')
args = parser.parse_args()

# GPU settings
gpu_device = args.gpu
if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(gpu_device).use()
xp = cuda.cupy if args.gpu >= 0 else np

data_file = args.data
n_epoch = args.epoch
character_embed_num=args.character_embed_num
feature_num = args.feature_num
hidden_num = args.hidden_num
batchsize = args.batchsize
testsize = args.testsize
vocabsize=args.vocabsize
training_rate=args.training_rate


def main():

    ###########################
    #### create dictionary ####
    ###########################

    corpus=Mimicdata()
    corpus.load("datapack_GloVe.pkl")


    ######################
    #### create model ####
    ######################

    model = Mimic(vocabsize, character_embed_num,feature_num=feature_num,hidden_num=hidden_num, batch_size=batchsize, gpu_flg=args.gpu,use_dropout=0)
    if args.gpu >= 0:
        model.to_gpu()
    optimizer = optimizers.Adam(alpha=training_rate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5))
    # optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    ##########################
    #### create ID corpus ####
    ##########################

    # separate corpus into Train and Test
    random.shuffle(corpus.data)
    test_data=corpus.data[0:0 + testsize]
    train_data=corpus.data[testsize:]

    #############################
    #### train mimic model ####
    #############################

    train_loss_data = []
    test_loss_data = []
    for num, epoch in enumerate(range(n_epoch)):
        total_loss = test_loss = 0
        batch_num = 0
        random.shuffle(train_data)

        # for training
        for i in range(0, len(corpus.data) - testsize, batchsize):

            # select batch data
            batch=train_data[i:i+batchsize]
            batch=list(zip(*batch))
            loss = model.encode(batch[0], batch[1], train=True)
            # learn model
            model.cleargrads()     # initialize all grad to zero
            loss.backward()  # back propagation
            optimizer.update()
            total_loss += float(loss.data)
            batch_num += 1
            print('Epoch: ', num, 'Batch_num', batch_num, 'batch loss: {:.2f}'.format(float(loss.data)))

        # for testing
        for i in range(0, testsize, batchsize):

            # select test batch data
            batch=test_data[i:i+batchsize]
            batch=list(zip(*batch))

            loss= model.encode(batch[0], batch[1], train=True)
            test_loss+=loss

        # save model and optimizer
        if (epoch + 1) % 5 == 0:
            print('-----', epoch + 1, ' times -----')
            print('save the model and optimizer')
            serializers.save_hdf5('data/glove_' + str(epoch) + '.model', model)
            serializers.save_hdf5('data/glove_' + str(epoch) + '.state', optimizer)

        # display the on-going status
        print('Epoch: ', num,
              'Train loss: {:.2f}'.format(total_loss),
              'Test loss: {:.2f}'.format(float(test_loss.data)))
        train_loss_data.append(float(total_loss / batch_num))
        test_loss_data.append(float(test_loss.data))

        # evaluate a test loss
        check_loss = test_loss_data[-10:]           # check out the last 10 loss data
        end_flg = [j for j in range(len(check_loss) - 1) if check_loss[j] < check_loss[j + 1]]

    # save loss data
    with open('./data/loss_train_data_glove.pkl', 'wb') as f:
        pickle.dump(train_loss_data, f)
    with open('./data/loss_test_data_glove.pkl', 'wb') as f:
        pickle.dump(test_loss_data, f)


if __name__ == "__main__":
    main()
