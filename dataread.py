import pickle
import codecs

with open("data/loss_test_data_w2v.pkl","rb") as f:
    data=pickle.load(f)

for i,d in enumerate(data):
    print(i,d)