import math
import time
from turtle import width
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import jieba
File = open("result/valid_ansphrases_9.txt","r+",encoding="utf-8")
wordlist = File.readlines()
phrases = [[],[],[],[]]
nums = np.zeros(shape=(4,))
sums = np.zeros(shape=(4,))
for x in wordlist:
    y = x.split(' ')
    c = int(float(y[1].strip()))
    phrases[len(y[0])-1].append((y[0],c))
    nums[len(y[0])-1] += 1
    sums[len(y[0])-1] += c

wFile = open("answord.txt","w+",encoding="utf-8")
phrases2 = []
for L in range(4):
    phrases[L].sort(key = lambda x:x[1]/sums[L],reverse=True)
    print(f"{L+1}字词")
    for i,x in enumerate(phrases[L]):
        if(i<10):
            print(x[0],x[1],x[1]/sums[L])
        phrases2.append((x[0],x[1],x[1]/sums[L]))
phrases2.sort(key=lambda x:x[2],reverse=True)
File = open("jiebadict.txt","r+",encoding="utf-8").readlines()
setw = set()
for x in File:
    setw.add(x.split(' ')[0])
cnt = 0
size = 0
for x in phrases2:
    print(x[0],x[1],x[2],file=wFile)
    if(1):
        size +=1
        if(x[0] in setw):
            cnt += 1
print("P:",cnt/size)

wFile.close()