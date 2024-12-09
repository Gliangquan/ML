import numpy as np
from sklearn.tree import DecisionTreeClassifier  # 用于创建决策树分类器的模块
from sklearn.datasets import make_classification  # 用于生成分类数据集
from sklearn.metrics import accuracy_score  # 用于计算模型的准确性
from sklearn import model_selection  # 导入数据划分和交叉验证模块

# 定义AdaBoost分类器类 AdaBoostClassifier的意思是 自适应增强分类器
class AdaBoostClassifier:
    def __init__(self, n_estimators=50):
        # 初始化AdaBoost分类器
        # param n_estimators: 弱分类器（决策树桩）的数量
        self.n_estimators = n_estimators  # 设置弱分类器数量
        self.alphas = []  # 存储每个弱分类器的权重
        self.models = []  # 存储弱分类器模型

    def fit(self, X, y):
        # 训练AdaBoost模型
        # param X: 训练样本特征
        # param y: 训练样本标签

        # 初始化样本权重（均匀分布）
        weights = np.ones(len(X)) / len(X)
        # 迭代训练多个弱分类器
        for _ in range(self.n_estimators):
            # 创建一个弱分类器，这里使用决策树桩（最大深度为1）
            model = DecisionTreeClassifier(max_depth=1)
            # 使用样本权重训练弱分类器
            model.fit(X, y, sample_weight=weights)
            # 预测训练集的标签
            predictions = model.predict(X)
            # 计算错误率：错分类的样本权重占比
            error = np.sum(weights * (predictions!= y)) / np.sum(weights)
            # 根据错误率计算弱分类器权重
            alpha = 0.5 * np.log((1 - error) / error)
            # 更新样本权重：增加被错分样本的权重，减小被正确分类样本的权重
            weights = weights * np.exp(-alpha * y * predictions)
            # 归一化权重，使其总和为1
            weights /= np.sum(weights)
            # 存储训练好的弱分类器和对应的权重
            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        # 使用AdaBoost模型预测标签
        # param X: 测试样本特征
        # return: 预测标签

        # 收集所有弱分类器的预测结果（每个分类器对所有样本的预测）
        predictions = np.array([model.predict(X) for model in self.models])
        alphas = np.array(self.alphas)
        # 根据权重对弱分类器的预测结果加权求和
        weighted_sum = np.sum(alphas.reshape(50, 1) * predictions.reshape(50, len(X)), axis=0)
        # 使用符号函数进行分类：正数预测为1，负数预测为-1
        return np.sign(weighted_sum).astype(int)


if __name__ == '__main__':
    # 生成一个二维分类数据集
    X, y = make_classification(n_samples=100,  # 样本数量
                               n_features=2,  # 特征数量
                               n_informative=2,  # 有效特征数量
                               n_redundant=0,  # 冗余特征数量
                               random_state=42)  # 随机种子

    # 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=1, test_size=0.2)

    # 创建AdaBoost分类器，并设置弱分类器数量为50
    adaboost = AdaBoostClassifier(n_estimators=50)
    # 使用训练集训练模型
    adaboost.fit(X_train, y_train)
    # 使用测试集进行预测
    y_pred = adaboost.predict(X_test)
    # 计算模型的准确性
    accuracy = accuracy_score(y_test, y_pred)
    # 输出AdaBoost模型的准确性
    print("AdaBoost准确性: {:.2f}".format(accuracy))