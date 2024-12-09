# 导入必要的库
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np  

# PCA 是一种降维技术，用于减少数据集的维度，同时保留尽可能多的信息。

# 加载数据集
digits = load_digits()
print("原始数据的维度:", digits['data'].shape)

# PCA降维到2维并可视化
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(digits['data'])
print("降维后的特征数量:", reduced_data.shape)

# 可视化降维后的数据
plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=digits.target,
                      edgecolor='none', alpha=0.5, cmap=plt.cm.viridis)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(scatter, label='Digit Label')
plt.title('PCA of Digits Dataset')
plt.show()

# 可视化数字的辅助函数
def plot_digits(data, title=""):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4), subplot_kw={'xticks': [], 'yticks': []})
    fig.suptitle(title)
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(data[i].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_clim(0, 16)

# 可视化原始无噪音的数字
plot_digits(digits.data, title="Original Digits")

# 添加高斯随机噪音
np.random.seed(42)
noisy = digits.data + np.random.normal(0, 4, digits.data.shape)
plot_digits(noisy, title="Noisy Digits")

# PCA降维去噪
pca = PCA(0.5)  # 保留95%的方差
pca.fit(noisy)
print("降维后保留的特征数量:", pca.n_components_)

# 利用逆变换重建数据
components = pca.transform(noisy)
filtered = pca.inverse_transform(components)
plot_digits(filtered, title="Denoised Digits")

plt.show()