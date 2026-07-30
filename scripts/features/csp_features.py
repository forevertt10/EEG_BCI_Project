import numpy as np
from mne.decoding import CSP


# =====================
# 读取数据
# =====================

X = np.load(
    "data/processed/A01T_X.npy"
)

y = np.load(
    "data/processed/A01T_y.npy"
)


print("原始数据:")
print(X.shape)



# =====================
# CSP
# =====================

csp = CSP(
    n_components=6,
    reg=None,
    log=True,
    norm_trace=False
)


X_csp = csp.fit_transform(
    X,
    y
)



print("----------------")

print("CSP特征:")
print(X_csp.shape)



# =====================
# 保存
# =====================

np.save(
    "data/processed/A01T_CSP.npy",
    X_csp
)


print("----------------")

print("CSP保存完成")