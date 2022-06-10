import math
import time
from turtle import width
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import jieba
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False
def ReadFile(path):
    File = open(path,"r+",encoding="utf-8")
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
    File.close()
    return nums,sums
X = np.arange(4)+1
nums,sums = ReadFile("result/valid_ansphrases_0.txt")
plt.subplot(1,2,1)
plt.bar(X-0.3,nums,width=0.3)

plt.subplot(1,2,2)
plt.bar(X-0.3,sums,width=0.3)

nums,sums = ReadFile("result/valid_ansphrases_5.txt")
plt.subplot(1,2,1)
plt.bar(X,nums,width=0.3)
plt.subplot(1,2,2)
plt.bar(X,sums,width=0.3)

nums,sums = ReadFile("result/valid_ansphrases_6.txt")
plt.subplot(1,2,1)
plt.bar(X+0.3,nums,width=0.3)
plt.title('不同字长词语数目')
plt.legend(['epoch 0','epoch 5','epoch 6'])

plt.subplot(1,2,2)
plt.bar(X+0.3,sums,width=0.3)
plt.title('不同字长词语频数')
plt.legend(['epoch 0','epoch 5','epoch 6'])
plt.savefig('show.png')