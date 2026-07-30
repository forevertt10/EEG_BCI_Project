import numpy as np
import os

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt



# =====================
# Load prediction
# =====================

y_pred = np.load(
    "results/logs/eegnet_preds.npy"
)

y_true = np.load(
    "results/logs/eegnet_true.npy"
)



# =====================
# Accuracy
# =====================

acc = accuracy_score(
    y_true,
    y_pred
)


print("----------------")
print("Accuracy:")
print(acc)



# =====================
# Classification report
# =====================

print("----------------")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "left_hand",
            "right_hand",
            "feet",
            "tongue"
        ]
    )
)



# =====================
# Confusion Matrix
# =====================

cm = confusion_matrix(
    y_true,
    y_pred
)


print("----------------")
print("Confusion Matrix:")
print(cm)



# =====================
# Save figure
# =====================

os.makedirs(
    "results/figures",
    exist_ok=True
)


plt.figure(
    figsize=(6,5)
)


plt.imshow(cm)


plt.title(
    "EEGNet Confusion Matrix"
)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "True"
)


plt.colorbar()


plt.savefig(
    "results/figures/eegnet_confusion_matrix.png",
    dpi=300
)


plt.close()


print("----------------")
print("Figure saved")