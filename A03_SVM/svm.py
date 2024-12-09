# SVM 分类鸢尾花数据集
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn import model_selection

# 获取数据
def get_data():
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target
    return x, y

# 获取分类器
def get_classifier():
    clf_linear = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
    clf_poly = svm.SVC(kernel='poly', degree=3, C=1).fit(x_train, y_train)
    clf_rbf = svm.SVC(kernel='rbf', gamma=1, C=1).fit(x_train, y_train)
    clf_rbf2 = svm.SVC(kernel='rbf', gamma=10, C=1).fit(x_train, y_train)
    return clf_linear, clf_poly, clf_rbf, clf_rbf2

# 训练和测试
def train_test(clf, i, x_train, x_test, y_train, y_test):
    # 训练模型
    clf.fit(x_train, y_train.ravel())  # ravel将多维数组转化为一维数组，需要加上括号调用函数
    # 计算准确率
    score = clf.score(x_test, y_test)
    print("准确率：", score)

# 绘制图像
def draw(clfs):
    x, y = get_data()
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))
    for clf, title, ax in zip(clfs, titles, axarr.flatten()):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3)
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)
        ax.set_title(title)
    plt.show()


if __name__ == '__main__':
    x, y = get_data()
    # 加载数据#划分训练集和测试集
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=0.2)
    # 将数据集分为训练集和测试集，测试集占总数据的 20%，random_state 确保每次划分相同
    clf_linear, clf_poly, clf_rbf, clf_rbf2 = get_classifier()  # 获取分类器

    # 存储分类器和对应标题的列表
    clfs = [clf_linear, clf_poly, clf_rbf, clf_rbf2]
    titles = ['Linear Kernel', 'polynomial Kernel with Degree=3',
              'Gaussian Kernel mith gamma=0.5', 'Gaussian Kernel with gamma=0.5']  # 分类器的描述标题
    # 绘制图像
    for i, clf in enumerate(clfs):
        train_test(clf, i, x_train, x_test, y_train, y_test)

    draw(clfs)