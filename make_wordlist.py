import pickle
import codecs

with codecs.open("D:/enwiki_vocab.txt","r","utf-8", errors='ignore')as f:
    lines=f.readlines() #len2176135

wordlist=[]
for line in lines:
    word=line.split(" ")[0]
    wordlist.append(word)

with open("wordlist_orderedbyfreq.pkl","wb") as f:
    pickle.dump(wordlist,f)