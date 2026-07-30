import numpy as np


X = np.load(
    "data/processed/A01T_X.npy"
)

y = np.load(
    "data/processed/A01T_y.npy"
)


print("X shape:")
print(X.shape)


print("----------------")


print("y shape:")
print(y.shape)


print("----------------")


print("类别统计:")


unique, counts = np.unique(
    y,
    return_counts=True
)


for u,c in zip(unique, counts):
    print(
        u,
        ":",
        c
    )