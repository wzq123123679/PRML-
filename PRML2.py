import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ==========================================
# 1. 核心数据生成函数 (基于图片代码)
# ==========================================
def make_moons_3d(n_samples=500, noise=0.1):
    # n_samples 是单个月亮的样本数。
    # 为了总共生成 2 * n_samples 个数据点
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t) 

    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    labels = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    X += np.random.normal(scale=noise, size=X.shape)
    
    return X, labels

# ==========================================
# 2. 生成训练集和测试集
# ==========================================
# 题目要求：训练集1000个数据 (分成两类，即n_samples=500)
X_train, y_train = make_moons_3d(n_samples=500, noise=0.2)

# 题目要求：测试集500个数据 (分成两类，即n_samples=250)
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)

# ==========================================
# 3. 定义并训练分类器
# ==========================================
# SVM选用三种核函数：Linear(线性), Poly(多项式), RBF(高斯径向基)
classifiers = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "AdaBoost (DT)": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=42),
    "SVM (Linear Kernel)": SVC(kernel='linear', random_state=42),
    "SVM (Poly Kernel)": SVC(kernel='poly', degree=3, random_state=42),
    "SVM (RBF Kernel)": SVC(kernel='rbf', gamma='scale', random_state=42)
}

# ==========================================
# 4. 测试与结果比较
# ==========================================
print("分类器在500个测试数据上的准确率表现：\n" + "-"*40)
for name, clf in classifiers.items():
    # 训练模型
    clf.fit(X_train, y_train)
    # 预测测试集
    y_pred = clf.predict(X_test)
    # 计算准确率
    acc = accuracy_score(y_test, y_pred)
    print(f"{name:<20}: {acc * 100:.2f}%")