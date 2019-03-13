import pandas as pd
import numpy as np
import nltk

f = open('stopwords.txt', 'r')
stopwords = f.read().splitlines()
f.close()


def removeStopwords(wordlist, stopwords):
    return [w for w in wordlist if w not in stopwords]

data_in = "dataset.csv"
print("importing file : "+data_in)
df = pd.read_csv(data_in, na_filter=False)

posR = df.loc[df['sentiment'] == 'positive']
negR = df.loc[df['sentiment'] == 'negative']


def getTopWords(frame):
    woerter = {}

    for index, row in frame.iterrows():
        print(index/frame.shape[0])
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

    sortedWords = sorted(woerter.items(), key=lambda kv: kv[1], reverse=True) 
        
    return sortedWords
    
top5pos = getTopWords(posR)[:100]
top5Neg = getTopWords(negR)[:100]


print("Top 100 words in positive reviews")
for la in top5pos:
    print(la)

print("Top 100 words in negative reviews")
for bla in top5Neg:
    print(bla)