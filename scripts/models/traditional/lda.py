import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import accuracy_score, classification_report



# =====================
# 加载CSP特征
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



print("----------------")

print("训练集:")
print(X_train.shape)

print("测试集:")
print(X_test.shape)



# =====================
# LDA模型
# =====================

lda = LinearDiscriminantAnalysis()



lda.fit(
    X_train,
    y_train
)



# =====================
# 测试
# =====================

y_pred = lda.predict(
    X_test
)



acc = accuracy_score(
    y_test,
    y_pred
)



print("----------------")

print("LDA Accuracy:")
print(acc)



print("----------------")

print(
    classification_report(
        y_test,
        y_pred
    )
)