# coding:utf-8
import numpy

#构建简单文本集,标签信息1表示侮辱性文档,0 表示正常文档
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1表示侮辱性文档,0 表示正常文档
    return postingList, classVec

# 统计词汇表，创建词汇表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    print(vocabSet)
    return list(vocabSet)

# 创建词频统计，此方法为词集模型，出现赋值1 ，未出现赋值0
def setOfWords2Vec(vocabList, inputSet):
    returnVec = []
    for article in inputSet:
        temp = [0] * len(vocabList)
        for word in article:
            if word in vocabList:
                temp[vocabList.index(word)] = 1 
            else:
                print("the word: %s is not in my Vocabulary!" % word)
        returnVec.append(temp)
    print(returnVec)
    return returnVec

# 词袋模型,统计概率
def bagOfWord2Vec(vocabList, inputSet):
    returnVec = []
    for article in inputSet:
        temp = [0] * len(vocabList)
        for word in article:
            if word in vocabList:
                temp[vocabList.index(word)] += 1
            else:
                print("the word: %s is not in my Vocabulary!" % word)
        returnVec.append(temp)
    print(returnVec)
    return returnVec

# 训练生成朴素贝叶斯
def trainNB(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs) # 统计侮辱性文档个数
    p0Num = numpy.ones(numWords)
    p1Num = numpy.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
            print(p1Denom,"p1Denom")
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
            print(p0Denom,"p0Denom")    
    p1Vect = numpy.log(p1Num / p1Denom)
    p0Vect = numpy.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

# 朴素贝叶斯分类器 参数2和参数3为训练集中侮辱性文档的概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = numpy.sum(vec2Classify * p1Vec) + numpy.log(pClass1)
    p0 = numpy.sum(vec2Classify * p0Vec) + numpy.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    # test = [['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him']]
    test = [['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid']]

    postingList, classVec = loadDataSet()
    vocabList = createVocabList(postingList)
    returnVec = bagOfWord2Vec(vocabList, postingList)
    p0Vect, p1Vect, pAbusive = trainNB(returnVec, classVec)
    print(pAbusive, p1Vect, p0Vect)
    testVec = bagOfWord2Vec(vocabList, test)
    pclass = classifyNB(testVec[0], p0Vect, p1Vect, pAbusive)  # 传入 testVec[0] 作为一维向量
    print(pclass)
    