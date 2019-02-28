import gensim
import pickle
import pandas as pd
import os
import argparse
import yaml
from scipy import stats
import numpy as np
from model import Interpreter


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--configfile', '-c', default="", type=str, help='')
args = parser.parse_args()
with open(args.configfile, "r+") as f:
    config = yaml.load(f)


interpreter = Interpreter(config)

file_path = "dataset/dir"
files = os.listdir(file_path)

columns=["vocab"]
datas=[]
data=[0]
for file in files:
    csv_data = pd.read_csv(file_path+file, names=('Word 1', 'Word 2', 'Human (mean)'))
    result = []
    drop_index = []
    f = 0
    for i, v in csv_data.iterrows():
       result.append(cos_sim(interpreter(v["Word 1"]), interpreter(v["Word 2"])))

    csv_data["similarate"]=result
    if f == 1:
        csv_data = csv_data[csv_data.similarate !="NaN"]

    corr = csv_data.corr(method='spearman')

    columns.append(file.rstrip(".txt.csv"))
    data.append(corr.loc['Human (mean)', "similarate"])
datas.append(data)

dataframe=pd.DataFrame(datas, columns=columns)

dataframe.to_csv(config["modelname"]+".csv")
