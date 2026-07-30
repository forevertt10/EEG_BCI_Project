import pandas as pd
import os


# =========================
# EEGNet repeated experiments
# =========================

runs = [
    0.7414,
    0.7586,
    0.6897,
    0.7931,
    0.6552
]


mean_acc = 0.7275862068965517
std_acc = 0.04925123054167483



# =========================
# Save
# =========================

os.makedirs(
    "results",
    exist_ok=True
)


df = pd.DataFrame(
    {
        "Experiment":
        [
            "Run1",
            "Run2",
            "Run3",
            "Run4",
            "Run5",
            "Mean",
            "Std"
        ],

        "Accuracy":
        runs + [
            mean_acc,
            std_acc
        ]
    }
)


df.to_csv(
    "results/eegnet_statistics.csv",
    index=False
)


print("EEGNet statistics saved")