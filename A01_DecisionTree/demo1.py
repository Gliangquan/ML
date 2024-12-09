from math import log
import operator

def calcShannonEnt(dataSet): # 计算数据集的熵
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
               ['短','粗','男'],
               ['长','细','女'],
               ['短','细','女'],
               ['短','粗','女'],
               ['长','粗','女'],
               ['长','粗','女']]
    labels = ['头发','声音']
    return dataSet, labels

def splitDataSet(dataSet, axis, value): # 按照特征划分数据集

    pass

def chooseBestFeatureToSplit(dataSet): # 选择最优特征
    pass
    

def majorityCnt(classList):
    pass


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
