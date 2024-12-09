from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# 加载Iris数据集
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 使用Bagging方法构建集成学习模型
bagging_model = BaggingClassifier(base_estimator=SVC(kernel='linear', probability=True), n_estimators=10, random_state=42, verbose=1)
bagging_model.fit(X_train, y_train)

# 使用随机森林方法构建集成学习模型
rf_model = RandomForestClassifier(n_estimators=10, random_state=42, verbose=1)
rf_model.fit(X_train, y_train)

# 在测试集上进行预测
bagging_pred = bagging_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# 计算并输出准确率
bagging_accuracy = accuracy_score(y_test, bagging_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"Bagging模型的准确率: {bagging_accuracy:.4f}")
print(f"随机森林模型的准确率: {rf_accuracy:.4f}")
