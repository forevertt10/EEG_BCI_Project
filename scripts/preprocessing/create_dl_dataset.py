import numpy as np


# ======================
# 加载预处理EEG
# ======================

X = np.load(
    "data/processed/A01T_X.npy"
)

y = np.load(
    "data/processed/A01T_y.npy"
)


print("Original X:")
print(X.shape)


print("Original y:")
print(y.shape)



# ======================
# 增加CNN输入维度
# ======================

X_dl = X[:, np.newaxis, :, :]


print("----------------")

print("Deep Learning X:")
print(X_dl.shape)



# ======================
# 保存
# ======================

np.save(
    "data/processed/A01T_DL_X.npy",
    X_dl
)


np.save(
    "data/processed/A01T_DL_y.npy",
    y
)


print("----------------")
print("Dataset saved")