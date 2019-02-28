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
import yaml
import chainer
import sys
from chainer import cuda, optimizers, serializers
from model import load_model


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def mse(v1, v2):
    return np.sum((v1-v2)**2)


def load_wordvector(config):
    wordvector_model = ""
    if config["wordvector"] == "word2vec":
        from gensim.models import KeyedVectors
        wordvector_model = KeyedVectors.load_word2vec_format(config["wordvectordir"], binary=False, unicode_errors="ignore")
    elif config["wordvector"] == "FastText":
        from gensim.models.wrappers import FastText
        wordvector_model = FastText.load_fasttext_format(config["wordvectordir"])

    assert wordvector_model != "", "wordvector is not defined"

    return wordvector_model


def calc_batch_loss(batch, config, model, wordvector_model):
    if config["use_relation"] == "Similarity":
        owv_list = []
        cwv_lists = []
        sim_lists = []
        for a in range(len(batch[1])):
            cw_l = batch[2][a][:config["compareword"] // 2]
            sim_l = batch[3][a][:config["compareword"] // 2]
            random_cw_list = random.sample(corpus.random_cw_list, config["compareword"] // 2)
            cw_l.extend(random_cw_list)
            for w in random_cw_list:
                sim_l.append(cos_sim(wordvector_model[batch[1][a]], wordvector_model[w]))
            cwv_l = []
            for cw in cw_l:
                cwv_l.append(wordvector_model[cw])
            owv_list.append(wordvector_model[batch[1][a]])
            cwv_lists.append(cwv_l)
            sim_lists.append(sim_l)

        return model.train(batch[0], owv_list, cwv_lists, sim_lists)

    elif config["use_relation"] == "Similarity_Random":
        owv_list = []
        cwv_lists = []
        sim_lists = []
        for a in range(len(batch[1])):
            cw_l = []
            sim_l = []
            random_cw_list = random.sample(corpus.random_cw_list, config["compareword"])
            cw_l.extend(random_cw_list)
            for w in random_cw_list:
                sim_l.append(cos_sim(wordvector_model[batch[1][a]], wordvector_model[w]))
            cwv_l = []
            for cw in cw_l:
                cwv_l.append(wordvector_model[cw])
            owv_list.append(wordvector_model[batch[1][a]])
            cwv_lists.append(cwv_l)
            sim_lists.append(sim_l)

        return model.train(batch[0], owv_list, cwv_lists, sim_lists)

    elif config["use_relation"] == "Similarity_Related":
        owv_list = []
        cwv_lists = []
        sim_lists = []
        for a in range(len(batch[1])):
            cw_l = batch[2][a][:config["compareword"]]
            sim_l = batch[3][a][:config["compareword"]]
            cwv_l = []
            for cw in cw_l:
                cwv_l.append(wordvector_model[cw])
            owv_list.append(wordvector_model[batch[1][a]])
            cwv_lists.append(cwv_l)
            sim_lists.append(sim_l)

        return model.train(batch[0], owv_list, cwv_lists, sim_lists)

    elif config["use_relation"] == "Normal":
        owv_list = []
        for a in range(len(batch[1])):
            owv_list.append(wordvector_model[batch[1][a]])

        return model.train(batch[0], owv_list)



def main():
    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--configfile', '-c', default="", type=str, help='')
    args = parser.parse_args()

    with open(args.configfile, "r+") as f:
        config = yaml.load(f)

    # GPU settings
    if config["GPU"] >= 0:
        cuda.check_cuda_available()
        cuda.get_device(config["GPU"]).use()
    xp = cuda.cupy if config["GPU"] >= 0 else np

    initial_embedding = ""
    if "init_emb" in config and config["init_emb"] != "None":
        with open(config["init_emb"], "rb") as f:
            initial_embedding = pickle.load(f)
    else:
        initial_embedding = None
    ######################
    #### create model ####
    ######################
    model, corpus = load_model(config, initial_embedding)

    wordvector_model = load_wordvector(config)

    if config["GPU"] >= 0:
        model.to_gpu()
    optimizer = optimizers.Adam(alpha=config["training_rate"])
    optimizer.setup(model)

    if "fix_embedding" in config and config["fix_embedding"]:
        model.enc.word_embed.disable_update()
    optimizer.add_hook(chainer.optimizer.GradientClipping(5))

    if config["NN_model"] in ["RNN", "GRU"]:
        corpus.train_data[0] = [xp.array(x, dtype=xp.int32) for x in corpus.train_data[0]]
        corpus.train_data = list(
            zip(corpus.train_data[0], corpus.train_data[1], corpus.train_data[2], corpus.train_data[3]))
        if hasattr(corpus, "dev_data"):
            corpus.dev_data[0] = [xp.array(x, dtype=xp.int32) for x in corpus.dev_data[0]]
            corpus.dev_data = list(zip(corpus.dev_data[0], corpus.dev_data[1], corpus.dev_data[2], corpus.dev_data[3]))
    elif config["NN_model"] in ["CNN", "SUM", "SUMFF"]:
        corpus.train_data[0] = [xp.array([x[i] if i < len(x) else -1 for i in range(corpus.max_input_len)],
                                dtype=xp.int32) for x in corpus.train_data[0]]
        corpus.train_data = list(
            zip(corpus.train_data[0], corpus.train_data[1], corpus.train_data[2], corpus.train_data[3]))
        if hasattr(corpus, "dev_data"):
            corpus.dev_data[0] = [xp.array([x[i] if i < len(x) else -1 for i in range(corpus.max_input_len)],
                                  dtype=xp.int32) for x in corpus.dev_data[0]]
            corpus.dev_data = list(zip(corpus.dev_data[0], corpus.dev_data[1], corpus.dev_data[2], corpus.dev_data[3]))
    else:
        print("model is not defined")
        exit()

    #############################
    #### train mimic model ####
    #############################

    if "overfit" in config and config["overfit"]:
        train_loss_data = []
        minimum_train_loss = 9999999
        minimum_epoch = 0
        minimum_train_loss_flag = 0
        for num, epoch in enumerate(range(999999)):
            total_loss = 0
            batch_num = 0
            random.shuffle(corpus.train_data)

            # for training
            for i in range(0, len(corpus.train_data), config["batchsize"]):

                # select batch data
                batch = corpus.train_data[i:i + config["batchsize"]]
                batch = list(zip(*batch))
                loss = calc_batch_loss(batch, config, model, wordvector_model)

                # learn model
                model.cleargrads()  # initialize all grad to zero
                loss.backward()  # back propagation
                optimizer.update()
                total_loss += float(loss.data)
                batch_num += 1
                # print('Epoch: ', num, 'Batch_num', batch_num, 'batch loss: {:.2f}'.format(float(loss.data)))

            # save model and optimizer
            if total_loss / batch_num < minimum_train_loss:
                print('-----', epoch + 1, ' times -----')
                print('save the model and optimizer for train loss')
                serializers.save_hdf5('data/' + config["modelname"] + '_best_train_loss.model', model)
                serializers.save_hdf5('data/' + config["modelname"] + '_best_train_loss.state', optimizer)
                minimum_train_loss = total_loss / batch_num
                minimum_epoch = epoch
                minimum_train_loss_flag = 0
            else:
                minimum_train_loss_flag += 1
                if minimum_train_loss_flag > 4:
                    break
            if epoch == 39:
                print('save the model and optimizer')
                serializers.save_hdf5('data/' + config["modelname"] + '_best.model', model)
                serializers.save_hdf5('data/' + config["modelname"] + '_best.state', optimizer)

            # display the on-going status
            print('Epoch: ', num,
                  'Train sim loss: {:.2f}'.format(total_loss))
            train_loss_data.append(float(total_loss / batch_num))

        # save loss data
        with open('./data/train_loss_' + config["modelname"] + '.pkl', 'wb') as f:
            pickle.dump(train_loss_data, f)
        print(minimum_epoch)

    else:
        train_loss_data = []
        dev_loss_data = []
        minimum_loss = 9999999
        minimum_train_loss = 9999999
        for num, epoch in enumerate(range(config["epoch"])):
            total_loss = dev_loss = 0
            batch_num = 0
            random.shuffle(corpus.train_data)

            # for training
            for i in range(0, len(corpus.train_data), config["batchsize"]):

                # select batch data
                batch = corpus.train_data[i:i+config["batchsize"]]
                batch = list(zip(*batch))
                loss = calc_batch_loss(batch, config, model, wordvector_model)

                # learn model
                model.cleargrads()  # initialize all grad to zero
                loss.backward()  # back propagation
                optimizer.update()
                total_loss += float(loss.data)
                batch_num += 1
                print('Epoch: ', num, 'Batch_num', batch_num, 'batch loss: {:.2f}'.format(float(loss.data)))

            # for developing
            for i in range(0, config["devsize"], config["batchsize"]):

                # select dev batch data
                batch = corpus.dev_data[i:i+config["batchsize"]]
                batch = list(zip(*batch))
                loss = calc_batch_loss(batch, config, model, wordvector_model)

                dev_loss += loss
            # save model and optimizer
            if dev_loss.data < minimum_loss:
                print('-----', epoch + 1, ' times -----')
                print('save the model and optimizer')
                serializers.save_hdf5('data/'+config["modelname"]+'_best.model', model)
                serializers.save_hdf5('data/'+config["modelname"]+'_best.state', optimizer)
                minimum_loss = dev_loss.data

            # save model and optimizer
            if total_loss / batch_num < minimum_train_loss:
                print('-----', epoch + 1, ' times -----')
                print('save the model and optimizer for train loss')
                serializers.save_hdf5('data/'+config["modelname"]+'_best_train_loss.model', model)
                serializers.save_hdf5('data/'+config["modelname"]+'_best_train_loss.state', optimizer)
                minimum_train_loss = total_loss / batch_num

            # display the on-going status
            print('Epoch: ', num,
                  'Train sim loss: {:.2f}'.format(total_loss),
                  'dev sim loss: {:.2f}'.format(float(dev_loss.data)))
            train_loss_data.append(float(total_loss / batch_num))
            dev_loss_data.append(float(dev_loss.data))

        # save loss data
        with open('./data/train_loss_' + config["modelname"] + '.pkl', 'wb') as f:
            pickle.dump(train_loss_data, f)
        with open('./data/dev_loss_' + config["modelname"] + '.pkl', 'wb') as f:
            pickle.dump(dev_loss_data, f)

        # evaluate with origin vector
        from trainedmodel_mimic_or_simmimic import interpreter
        interpreter = interpreter(config)

        mse_total = 0
        cos_sim_total = 0
        total = 0
        for word in corpus.test_data:
            v_o = wordvector_model[word]
            v_m = interpreter(word)
            mse_total += mse(v_o, v_m)
            cos_sim_total += cos_sim(v_o, v_m)
            total += 1

        print(mse_total/total/config["feature_num"])
        print(cos_sim_total/total)

if __name__ == "__main__":
    main()
