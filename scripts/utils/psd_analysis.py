import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


# =====================
# 加载数据
# =====================

X = np.load(
    "data/processed/A01T_X.npy"
)


y = np.load(
    "data/processed/A01T_y.npy"
)



# =====================
# 选择trial
# =====================

trial = 0


# 选择C3附近通道
channel = 7


signal = X[trial, channel]



# =====================
# PSD计算
# =====================

sfreq = 250


freqs, psd = welch(
    signal,
    fs=sfreq,
    nperseg=256
)



# =====================
# 绘图
# =====================

plt.figure(figsize=(10,5))


plt.plot(
    freqs,
    psd
)


plt.xlim(
    0,
    40
)


plt.xlabel(
    "Frequency (Hz)"
)


plt.ylabel(
    "Power"
)


plt.title(
    "EEG Power Spectral Density"
)


plt.grid()


plt.tight_layout()


plt.savefig(
    "results/figures/PSD_example.png",
    dpi=300
)


plt.show()