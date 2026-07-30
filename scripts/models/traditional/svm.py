import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report



# =====================
# 加载数据
# =====================

X = np.load(
    "data/processed/A01T_CSP.npy"
)

y = np.load(
    "data/processed/A01T_y.npy"
)



print("Feature shape:")
print(X.shape)



# =====================
# 数据划分
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# =====================
# SVM模型
# =====================

svm = SVC(
    kernel="rbf",
    C=1,
    gamma="scale"
)


svm.fit(
    X_train,
    y_train
)



# =====================
# 测试
# =====================

y_pred = svm.predict(
    X_test
)



acc = accuracy_score(
    y_test,
    y_pred
)


print("----------------")

print("SVM Accuracy:")

print(acc)



print("----------------")

print(
    classification_report(
        y_test,
        y_pred
    )
)