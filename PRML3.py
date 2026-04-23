import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
df = pd.read_csv('LSTM-Multivariate_pollution.csv')

# 2. 数据查看（已修复 pm2.5 → pollution）
print("=== 数据集形状（行×列）===")
print(f"{df.shape[0]} 行 × {df.shape[1]} 列\n")

print("=== 前5行数据预览 ===")
print(df.head())

print("\n=== 数据类型与缺失值统计 ===")
print(df.info())

print("\n=== 数据基本统计描述 ===")
print(df.describe().round(2))

print("\n=== 目标变量（pollution）缺失值占比 ===")
pollution_missing_rate = df['pollution'].isnull().sum() / len(df) * 100
print(f"缺失值占比：{pollution_missing_rate:.2f}%")

# ------------------------------------------------------------------------------
# 3. 数据预处理
# ------------------------------------------------------------------------------
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# 风向编码
df_encoded = pd.get_dummies(df, columns=['wnd_dir'], prefix='wnd_dir')

# 分离特征和目标
target_col = 'pollution'
features = df_encoded.drop(columns=[target_col])
target = df_encoded[target_col]

# 标准化
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))

features_scaled = scaler_features.fit_transform(features)
target_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1))

# 构建时序数据
def create_sequences(features, target, time_steps=24):
    X, y = [], []
    for i in range(time_steps, len(features)):
        X.append(features[i-time_steps:i, :])
        y.append(target[i, 0])
    return np.array(X), np.array(y)

time_steps = 24
X, y = create_sequences(features_scaled, target_scaled, time_steps)

# 划分训练集测试集
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("\n=== 数据格式准备完成 ===")
print(f"训练集X: {X_train.shape}")
print(f"测试集X: {X_test.shape}")

# ------------------------------------------------------------------------------
# 4. 构建LSTM模型
# ------------------------------------------------------------------------------
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 训练
print("\n=== 开始训练 ===")
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    shuffle=False
)

# ------------------------------------------------------------------------------
# 5. 预测与评估
# ------------------------------------------------------------------------------
y_pred_scaled = model.predict(X_test, verbose=0)
y_pred = scaler_target.inverse_transform(y_pred_scaled)
y_test_original = scaler_target.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))

print("\n=== 模型评估结果 ===")
print(f"平均绝对误差 MAE: {mae:.2f}")
print(f"均方根误差 RMSE: {rmse:.2f}")

# ------------------------------------------------------------------------------
# 6. 画图
# ------------------------------------------------------------------------------
plt.figure(figsize=(12,5))
plt.plot(y_test_original[:100], label='真实值')
plt.plot(y_pred[:100], label='预测值')
plt.title('PM2.5预测结果对比')
plt.legend()
plt.show()