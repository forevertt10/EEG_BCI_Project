import os
import pandas as pd
import matplotlib.pyplot as plt



# ==========================
# Model results
# ==========================

models = [
    "LDA+CSP",
    "SVM+CSP",
    "Random Forest+CSP",
    "CNN",
    "EEGNet"
]


accuracy = [
    0.6724,
    0.7586,
    0.7759,
    0.4310,
    0.7241
]



# ==========================
# Save CSV
# ==========================

os.makedirs(
    "results",
    exist_ok=True
)


df = pd.DataFrame(
    {
        "Model": models,
        "Accuracy": accuracy
    }
)


df.to_csv(
    "results/model_comparison.csv",
    index=False
)


print("----------------")
print("CSV saved")



# ==========================
# Plot
# ==========================

os.makedirs(
    "results/figures",
    exist_ok=True
)


plt.figure(
    figsize=(8,5)
)


plt.bar(
    models,
    accuracy
)


plt.ylabel(
    "Accuracy"
)


plt.xlabel(
    "Model"
)


plt.title(
    "Comparison of Machine Learning and Deep Learning Methods"
)


plt.xticks(
    rotation=45,
    ha="right"
)


for i,v in enumerate(accuracy):

    plt.text(
        i,
        v+0.01,
        f"{v:.2f}",
        ha="center"
    )


plt.tight_layout()


plt.savefig(
    "results/figures/model_comparison.png",
    dpi=300
)


plt.close()


print("----------------")
print("Figure saved")