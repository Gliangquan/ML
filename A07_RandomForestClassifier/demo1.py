from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# 随机森林分类器
if __name__ == "__main__":
    # 加载鸢尾花数据集
    dataset = load_iris()
    X = dataset.data
    y = dataset.target

    # 将数据集随机划分为训练集和测试集
    # 默认划分比例是75%训练集，25%测试集
    # random_state=14：随机种子数，相同的种子数，每次划分的结果是一样的
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)

    # 创建随机森林分类器实例
    clf = RandomForestClassifier(max_depth=2, random_state=0)

    # 使用训练集数据拟合（训练）随机森林分类器
    clf = clf.fit(X_train, y_train)

    # 使用训练好的模型对测试集数据进行预测
    y_predicted = clf.predict(X_test)

    # 计算预测的准确率：测试集中预测正确的样本比例
    accuracy = np.mean(y_predicted == y_test) * 100

    # 打印测试集的真实标签
    print('y_test', y_test)

    # 打印模型预测的标签
    print('y_predicted', y_predicted)

    # 打印预测的准确率
    print('accuracy', accuracy)