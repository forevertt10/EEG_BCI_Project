import numpy as np
import matplotlib.pyplot as plt


# =====================
# 加载数据
# =====================

X = np.load(
    "data/processed/A01T_X.npy"
)


y = np.load(
    "data/processed/A01T_y.npy"
)


print(X.shape)


# =====================
# 选择一个trial
# =====================

trial = 0


eeg = X[trial]


# =====================
# 通道选择
# =====================

# 你的22个通道顺序来自MNE
# 需要先查看真实顺序

print(eeg.shape)


# =====================
# 绘制前三个通道
# =====================

plt.figure(figsize=(12,6))


for i in range(3):

    plt.plot(
        eeg[i],
        label=f"Channel {i}"
    )


plt.title(
    f"EEG Trial {trial}, Label={y[trial]}"
)


plt.xlabel(
    "Samples"
)


plt.ylabel(
    "Normalized amplitude"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    "results/figures/trial_example.png",
    dpi=300
)


plt.show()