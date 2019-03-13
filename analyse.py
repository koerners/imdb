import pandas as pd
import numpy as np
import nltk

f = open('stopwords.txt', 'r')
stopwords = f.read().splitlines()
f.close()
# print(stopwords)

woerter = {}
woerter1 = {}

def removeStopwords(wordlist, stopwords):
    return [w for w in wordlist if w not in stopwords]

data_in = "dataset.csv"
print("importing file : "+data_in)
df = pd.read_csv(data_in, na_filter=False)

posR = df.loc[df['sentiment'] == 'positive']
negR = df.loc[df['sentiment'] == 'negative']

for index, row in posR.iterrows():
    print(index/posR.shape[0])
    r = row['review']
    words = nltk.word_tokenize(r)   
    for word in words:
        word = word.lower()
        if word in woerter: 
            woerter[word]+=1
        else:
            woerter[word]=1

for key in woerter:
    if key in stopwords:
        woerter[key]=0

sortedWordsPos = sorted(woerter.items(), key=lambda kv: kv[1], reverse=True) 
top5pos = sortedWordsPos[:50]


for index, row in negR.iterrows():
    print(index/negR.shape[0])
    r = row['review']
    words = nltk.word_tokenize(r)   
    for word in words:
        word = word.lower()
        if word in woerter1: 
            woerter1[word]+=1
        else:
            woerter1[word]=1

for key in woerter1:
    if key in stopwords:
        woerter1[key]=0

sortedWordsNeg = sorted(woerter1.items(), key=lambda kv: kv[1], reverse=True) 
top5Neg = sortedWordsNeg[:50]


print("Top 50 words in positive reviews")
for la in top5pos:
    print(la)

print("Top 50 words in negative reviews")
for bla in top5Neg:
    print(bla)