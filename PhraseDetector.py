# powered by Zhaojun Huang
# 2022.6.7

import sys
import math
import time
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def is_Chinese(uchar):
    if (ord(uchar) >= int('0x4e00', 16)) and (ord(uchar) <= int('0x9fa5', 16)):
        return True
    else:
        return False


def ID(word):
    if(word not in index):
        return -1
    return index[word][0]

def TEXT(id):
    return phrases[id][0]

def minPMI(wid):
    global possibility
    pmi = 1000000000
    pkl = 1000000000
    word = TEXT(wid)
    L = len(word)
    for i in range(1,L+1):
        pmi = min(pmi,math.log(possibility[wid]/possibility[ID(word[:i])]/possibility[ID(word[i:])]))
        pkl = min(pkl,possibility[wid]*math.log(possibility[wid]/possibility[ID(word[:i])]/possibility[ID(word[i:])]))
    return pmi,pkl

def TrainClassifier(Sample_words, Sample_nowords):
    #Sample_words save ID
    global Sec
    global frequency
    global possibility
    global counts
    global prenums
    sys.stderr.write('Use %d seconds. Begin to Training Classifier.\n' %
                     (time.time()-Sec)) 
    Sample_size = len(Sample_words)+len(Sample_nowords)
    Samples = np.zeros(shape=(Sample_size,5))
    Targets = np.zeros(shape=(Sample_size,))
    L = len(Sample_words)
    for i,wid in enumerate(Sample_words):
        Samples[i][0] = frequency[wid]
        Samples[i][1] = possibility[wid]
        Samples[i][2],Samples[i][3] = minPMI(wid)
        L = len(TEXT(wid))
        Samples[i][4] = counts[L-1][wid-prenums[L-1]]
        Targets[i] = 1
    for j,wid in enumerate(Sample_nowords):
        i = j + L
        Samples[i][0] = frequency[wid]
        Samples[i][1] = possibility[wid]
        Samples[i][2],Samples[i][3] = minPMI(wid)
        L = len(TEXT(wid))
        Samples[i][4] = counts[L-1][wid-prenums[L-1]]
        Targets[i] = 0
    Classifier = RandomForestClassifier(n_estimators=25)
    Classifier.fit(Samples, Targets)
#	sys.stderr.write('Use %d seconds. Score on training samples : '+str(Classifier.score(Samples[Len:], Targets[Len:])) + '.\n')
#	sys.stderr.write('Use %d seconds. Score on testing samples : '+str(Classifier.score(Samples[:Len], Targets[:Len])) + '.\n')
    return Classifier

def ClassifierPredict(Classifier,RG):
    global Sec
    global frequency
    global possibility
    global counts
    global totnum
    global prenums
    sys.stderr.write('Use %d seconds. Begin to Predict quality .\n' %
                     (time.time()-Sec)) 
    Features = np.zeros(shape=(totnum,5))
    for wid in RG:
        Features[wid][0] = frequency[wid]
        Features[wid][1] = possibility[wid]
        Features[wid][2],Features[wid][3] = minPMI(wid)
        L = len(TEXT(wid))
        Features[wid][4] = counts[L-1][wid-prenums[L-1]]
    Result = Classifier.predict_proba(Features)
    sys.stderr.write('Use %d seconds. Quality Predicted.\n' %
                     (time.time()-Sec)) 
    return Result

path = "E:/pku/2022_Spring/程序设计思维/project/"

# reading phrases
mode = "valid"
Sec = time.time()
Sentences = open(path+mode+"_sentences.txt", "r+", encoding="utf-8").readlines()
Sentences = list(map(lambda x:x.strip(),Sentences))
phrases = [(x.split(' ')[0], int(x.split(' ')[1])) for x in (
    open(path+mode+"_fq_phrases.txt", "r+", encoding="utf-8").readlines())]
index = dict([(x[0], (i, len(x[0])-1)) for i, x in enumerate(phrases)])

nums = np.zeros(shape=(4,), dtype=np.int32)
sums = np.zeros(shape=(4,), dtype=np.int32)
sys.stderr.write('Use %d seconds. Begin to calculate num/sum .\n' %
                     (time.time()-Sec)) 
for x in phrases:
    nums[len(x[0])-1] += 1
    sums[len(x[0])-1] += x[1]

prenums = np.array([nums[i:].sum() for i in range(6)])
totnum = len(phrases)
totsum = sums.sum()
frequency = np.array([x[1]/sums[len(x[0])-1] for x in phrases])
possibility = np.array([x[1]/totsum for x in phrases])
# each phrase's quality equal to 1 at first
counts = np.zeros(shape=(4,nums.max()))

for wid,x in enumerate(phrases):
    L = len(x[0])
    counts[L-1][wid - prenums[L-1]] = frequency[wid]


# loading the wordlist,nowordlist
Sample_words = []
Sample_nowords = []
for x in open("sample_wordlist.txt","r+",encoding="utf-8").readlines():
    y = x.strip()
    if (y in index):
        Sample_words.append(ID(y))
for x in open("sample_nowordlist.txt","r+",encoding="utf-8").readlines():
    y = x.strip()
    if (y in index):
        Sample_nowords.append(ID(y))

# Estimate alpha, using binary search
"""
l = 0
r = 1
rate = 0.98
usewordlist = []
useworddict = {}
for sid in Sample_words:
    sentence = TEXT(sid)
    L = len(sentence)
    for i in range(1,L+1):
        for j in reversed(range(i)):
            wid = ID(sentence[j:i])
            usewordlist.append(wid)
            useworddict[wid] = len(usewordlist)
while (r-l>0.00000001):
    alpha = (l+r)/2
    lenpenalty = [math.log(alpha ** (x)) for x in range(0, 6)]
    print(alpha)
    Classifier = TrainClassifier(Sample_words,Sample_nowords)
    cnt = 0
    quality = ClassifierPredict(Classifier,usewordlist)
    for sid in Sample_words:
        sentence = TEXT(sid)
        L = len(sentence)
        f = np.zeros(L+1)
        g = np.zeros(L+1).astype('int32')
        g[:] = -1
        f[0] = 0
        for i in range(1,L+1):
            for j in reversed(range(i)):
                wid = ID(sentence[j:i])
                if(wid==-1):
                    break
                qv = quality[useworddict[wid]][1]
                if(frequency[wid]!=0 and qv!=0):
                    v = f[j] + math.log(frequency[wid])+math.log(qv)+lenpenalty[i-j-1]
                if(g[i] ==-1 or f[i]<v):
                    f[i] = v
                    g[i] = j
        if(g[L] == 0):
            cnt+=1
    print(cnt)
    size = len(Sample_words)
    if(cnt<size):
        r = alpha
    else:
        l = alpha
"""
#alpha = 0.02
#lenpenalty = [math.log(alpha ** (x)) for x in range(0, 6)]
#print(alpha)
lenpenalty = [0.036,0.64,0.16,0.14]

# Iterative process, we loop for 10 epoch
for epoch in range(20):
    print(f"Epoch {epoch}")
    # updating classifier (quality estimation (feed back))
    Classifier = TrainClassifier(Sample_words,Sample_nowords)
    quality = ClassifierPredict(Classifier,range(totnum))
    print(quality)
    saved_counts = counts.copy()
    counts[:,:] = 1
    ncounts = nums.copy()
    # phrasal segmentation (Viterbi Training)
    for sentence in Sentences:
        L = len(sentence)
        f = np.zeros(L+1)
        g = np.zeros(L+1).astype('int32')
        g[:] = -1
        f[0] = 0
        
        for i in range(1,L+1):
            for j in reversed(range(i)):
                wid = ID(sentence[j:i])
                if(wid==-1):
                    break
                if(frequency[wid]!=0 and quality[wid][1]!=0):
                    v = f[j] + math.log(saved_counts[i-j-1][wid-prenums[i-j-1]])+math.log(quality[wid][1])+lenpenalty[i-j-1]
                if(g[i] ==-1 or f[i]<v):
                    f[i] = v
                    g[i] = j
        cur = L
        while(cur!=0):
            if(g[cur] == -1):
                break
            wid = ID(sentence[g[cur]:cur])
            if(wid==-1):
                break
            counts[cur-g[cur]-1][wid - prenums[cur-g[cur]-1]] += 1
            ncounts[cur-g[cur]-1] += 1
            cur = g[cur]
    # save the result
    wFile =  open("result/"+mode+f"_ansphrases_{epoch}.txt","w+",encoding="utf-8")
    ans_phrases = []
    for wid,x in enumerate(phrases):
        L = len(x[0])
        if(counts[L-1][wid - prenums[L-1]]>1):
            print(x[0],counts[L-1][wid - prenums[L-1]],file=wFile)
    wFile.close()
    # update new theta
    counts = (counts.transpose(1,0)/ncounts).transpose(1,0)