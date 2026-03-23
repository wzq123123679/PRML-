import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

train_df = pd.read_excel("Data4Regression.xlsx", sheet_name=0)
test_df = pd.read_excel("Data4Regression.xlsx", sheet_name=1)

x_train = train_df['x'].values
y_train = train_df['y_complex'].values
x_test = test_df['x_new'].values
y_test = test_df['y_new_complex'].values

X_train = np.column_stack((np.ones_like(x_train), x_train))
X_test = np.column_stack((np.ones_like(x_test), x_test))

w_ls = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

def gradient_descent(X, y, lr=0.01, max_iter=10000, tol=1e-6):
    w = np.zeros(X.shape[1])
    for _ in range(max_iter):
        grad = X.T @ (X @ w - y) / len(y)
        w_new = w - lr * grad
        if np.linalg.norm(w_new - w) < tol:
            break
        w = w_new
    return w
w_gd = gradient_descent(X_train, y_train, lr=0.01)

def newton_method(X, y, max_iter=100, tol=1e-6):
    w = np.zeros(X.shape[1])
    H = X.T @ X / len(y) 
    for _ in range(max_iter):
        grad = X.T @ (X @ w - y) / len(y)
        w_new = w - np.linalg.inv(H) @ grad
        if np.linalg.norm(w_new - w) < tol:
            break
        w = w_new
    return w
w_newton = newton_method(X_train, y_train)

degree = 5
X_poly_train = np.column_stack([x_train**d for d in range(degree+1)])
X_poly_test = np.column_stack([x_test**d for d in range(degree+1)])
w_poly = np.linalg.inv(X_poly_train.T @ X_poly_train) @ X_poly_train.T @ y_train

x_plot = np.linspace(min(x_train.min(), x_test.min()), max(x_train.max(), x_test.max()), 200)

y_ls_plot = w_ls[0] + w_ls[1] * x_plot
y_gd_plot = w_gd[0] + w_gd[1] * x_plot
y_newton_plot = w_newton[0] + w_newton[1] * x_plot

X_poly_plot = np.column_stack([x_plot**d for d in range(degree+1)])
y_poly_plot = X_poly_plot @ w_poly

plt.figure(figsize=(12, 8))

plt.scatter(x_train, y_train, label='训练数据', color='lightblue', alpha=0.7, s=50)
plt.scatter(x_test, y_test, label='测试数据', color='orange', alpha=0.7, s=50, marker='^')

plt.plot(x_plot, y_ls_plot, 'r-', label='线性-最小二乘法', linewidth=2)
plt.plot(x_plot, y_gd_plot, 'g--', label='线性-梯度下降法', linewidth=2, alpha=0.8)
plt.plot(x_plot, y_newton_plot, 'b-.', label='线性-牛顿法', linewidth=2, alpha=0.8)
plt.plot(x_plot, y_poly_plot, 'purple', label='5次多项式回归', linewidth=2)

plt.xlabel('x 自变量', fontsize=12)
plt.ylabel('y 因变量', fontsize=12)
plt.title('四种拟合方法效果对比', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()

y_test_ls = X_test @ w_ls
y_test_gd = X_test @ w_gd
y_test_newton = X_test @ w_newton
y_test_poly = X_poly_test @ w_poly

mse_ls = np.mean((y_test - y_test_ls)**2)
mse_gd = np.mean((y_test - y_test_gd)**2)
mse_newton = np.mean((y_test - y_test_newton)**2)
mse_poly = np.mean((y_test - y_test_poly)**2)

print(f"ls_mse:{mse_ls:.4f}")
print(f"gd_mse:{mse_gd:.4f}")
print(f"newton_mse:{mse_newton:.4f}")
print(f"poly_mse:{mse_poly:.4f}")
