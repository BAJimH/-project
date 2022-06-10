import collections
import math
import time
import json

path = "E:/pku/2022_Spring/程序设计思维/project/"
# To find those Phrases which appear frequently


def is_Chinese(uchar):
    if (ord(uchar) >= int('0x4e00',16)) and (ord(uchar)<=int('0x9fa5',16)):
        return True
    else:
        return False

def FindFrequentPhrase(Corpus, Punctuation, Threshold):
    Frequency = collections.defaultdict(int)
    index = collections.defaultdict(set)
    length = len(Corpus)
    for i in range(length):
        if (Corpus[i] not in Punctuation) and is_Chinese(Corpus[i]):
            index[Corpus[i]].add(i)
    L = 1
    while len(index):
        index2 = collections.defaultdict(set)
        print(L)
        L +=1
        if(L>5):
            break
        for now in index:
            if (len(index[now]) >= Threshold):
                Frequency[now] = len(index[now])
                for p in index[now]:
                    if (p+1 < length) and (Corpus[p+1] not in Punctuation) and is_Chinese(Corpus[p+1]):
                        nxt = now+Corpus[p+1]
                        index2[nxt].add(p+1)
        index = index2
    return Frequency


Punctuation = {"-", "—", "(", ")", ",", ".", "=", "。", "？", "！", "，", "“", "”", "（", "）", ";", "；", ":", "：", "<", "《", "》", ">", " ", "\n",
               "─", "Ⅱ", "~" ,"◆" ,"【","】","_","　", "―", "・ ", "%", "、", "・ ", "・", "Ⅲ", "Ⅰ", "/", "!", "…", "?", "", "’", "‘", "━", "〈", "〉", "]", "[", "+", "/", "\""}


def main():
    Sec = time.time()
    # open file
    File = open(path+"news2016zh_valid.json", "r+", encoding="utf-8")
    DataList = File.readlines()
    # make puntuation set
    for i in range(10):
        Punctuation.add(str(i))
    for i in range(ord("A"), ord("Z")+1):
        Punctuation.add(str(chr(i)))
    for i in range(ord("a"), ord("z")+1):
        Punctuation.add(str(chr(i)))
        # making the corpus
    CorpusList = []
    for i, line in enumerate(DataList):
        if(i % 100 == 0):
            print(i)
        Jfile = json.loads(line)
        for x in Jfile.values():
            CorpusList.append(x)
    Corpus = "\n".join(CorpusList)
    print("Begin to find phrase") 
    Frequency = FindFrequentPhrase(Corpus, Punctuation, 5)
    Frequency = dict(
        sorted(Frequency.items(), key=lambda item: len(item[0]), reverse=True))

    Filew = open(path+"valid_fq_phrases.txt", 'w+', encoding='utf-8')
    Filew2 = open(path+"valid_Corpus.txt", 'w+', encoding='utf-8')
    Frequency = dict(
        sorted(Frequency.items(), key=lambda item: len(item[0]), reverse=True))

    for i in Frequency.keys():
        Filew.write(str(i)+" "+str(Frequency[i])+"\n")
    Filew.close()
    Filew2.write(Corpus)
    Filew2.close()
    print("Use %d seconds" % (time.time()-Sec))


if __name__ == "__main__":
    main()
