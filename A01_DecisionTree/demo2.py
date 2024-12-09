from math import log
import operator

def calcShannonEnt(dataSet):
    numEntries = len(dataSet) # 数据集的行数
    labelCounts = {} # 存储每个标签出现的次数
    for featVec in dataSet: # 遍历数据集
        currentLabel = featVec[-1] # 获取当前样本的标签
        if currentLabel not in labelCounts.keys(): # 如果当前标签不在labelCounts中
            labelCounts[currentLabel] = 0 # 初始化当前标签的计数
        labelCounts[currentLabel] += 1 # 当前标签的计数加1
    shannonEnt = 0.0 # 初始化熵
    for key in labelCounts: # 遍历labelCounts
        prob = float(labelCounts[key])/numEntries # 计算当前标签的概率
        shannonEnt -= prob * log(prob, 2) # 计算当前标签的熵
    return shannonEnt # 返回熵

# 创建数据集
def createDataSet1():
    dataSet = [['长','粗','男'],
               ['短','粗','男'],
               ['短','细','女'],
               ['长','细','女'],
               ['长','粗','女'],
               ['短','粗','女'],
               ['长','细','男'],
               ['短','细','男'],
               ['长','粗','男'],
               ['短','粗','男']]
    labels = ['头发','声音']
    return dataSet, labels

def splitDataSet(dataSet, axis, value): 
    retDataSet = [] # 存储划分后的数据集
    for featVec in dataSet: # 遍历数据集
        if featVec[axis] == value: # 如果当前样本的特征值等于value
            reducedFeatVec = featVec[:axis] # 获取当前样本的特征值
            reducedFeatVec.extend(featVec[axis+1:]) # 获取当前样本的特征值
            retDataSet.append(reducedFeatVec) # 将当前样本添加到划分后的数据集中
    return retDataSet # 返回划分后的数据集

# 选择最优特征
def chooseBestFeatureToSplit(dataSet): 
    numFeatures = len(dataSet[0]) - 1 # 特征的数量
    baseEntropy = calcShannonEnt(dataSet) # 计算数据集的熵
    bestInfoGain = 0.0 # 初始化信息增益
    bestFeature = -1 # 初始化最优特征
    for i in range(numFeatures): # 遍历特征
        featList = [example[i] for example in dataSet] # 获取当前特征的所有取值
        uniqueVals = set(featList) # 获取当前特征的所有取值的集合
        newEntropy = 0.0 # 初始化新的熵
        for value in uniqueVals: # 遍历当前特征的所有取值
            subDataSet = splitDataSet(dataSet, i, value) # 按照当前特征划分数据集
            prob = len(subDataSet)/float(len(dataSet)) # 计算当前特征取值的概率
            newEntropy += prob * calcShannonEnt(subDataSet) # 计算当前特征取值的熵
        infoGain = baseEntropy - newEntropy # 计算信息增益
        if (infoGain > bestInfoGain): # 如果当前信息增益大于最优信息增益
            bestInfoGain = infoGain # 更新最优信息增益
            bestFeature = i # 更新最优特征
    return bestFeature # 返回最优特征

# 选择出现次数最多的标签
def majorityCnt(classList): 
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 创建决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet] # 获取数据集的标签
    if classList.count(classList[0]) == len(classList): # 如果所有样本的标签都相同
        return classList[0] # 返回该标签
    if len(dataSet[0]) == 1: # 如果所有特征都已经被使用
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) # 选择最优特征
    bestFeatLabel = labels[bestFeat] # 获取最优特征的标签
    myTree = {bestFeatLabel:{}} # 创建决策树
    del(labels[bestFeat]) # 删除最优特征的标签
    featValues = [example[bestFeat] for example in dataSet] # 获取最优特征的所有取值
    uniqueVals = set(featValues) # 获取最优特征的所有取值的集合
    for value in uniqueVals: # 遍历最优特征的所有取值
        subLabels = labels[:] # 复制标签
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels) # 递归创建决策树
    return myTree # 返回决策树

if __name__ == '__main__':
    dataSet, labels = createDataSet1() # 创建数据集
print(createTree(dataSet, labels)) # 创建决策树
