import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体为微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


# 加载数据集
np.random.seed(5)  # 为了可重复性，设置随机种子
iris = datasets.load_iris()
# 提取数据
X = iris.data
y = iris.target

# 训练KMeans模型
clf = KMeans(n_clusters=3)  # n_clusters=3表示将数据分为3个簇
clf.fit(X)  # 拟合数据
labels = clf.labels_  # 获取簇标签

# 可视化结果
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d', elev=48, azim=134)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(float), edgecolor='k')

# 隐藏坐标轴刻度
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

# 设置坐标轴标签和标题
ax.set_xlabel('petal length')
ax.set_ylabel('sepal length')
ax.set_zlabel('petal length')
ax.set_title('3')

plt.show()