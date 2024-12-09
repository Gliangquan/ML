import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# SVD.py 文件
# SVD (Singular Value Decomposition) 是一种矩阵分解技术，用于降维、去噪和特征提取。

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target
print("数据维度:", X.shape)

# SVD 分解
U, s, V = np.linalg.svd(X)
print("U 矩阵的维度:", U.shape)
newdata = U[:, :2] # 取前两个奇异向量来降维

# 可视化
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
for label in np.unique(y):
    ax.scatter(newdata[y == label, 0], newdata[y == label, 1], label=f"Class {label}")
plt.xlabel('SVD1')
plt.ylabel('SVD2')
plt.title("SVD Visualization of Iris Dataset")
plt.legend()
plt.show()