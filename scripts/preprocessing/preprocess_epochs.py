import mne
import numpy as np


file_path = "data/raw/BCICIV_2a_gdf/A01T.gdf"


# =====================
# 1. 读取数据
# =====================

raw = mne.io.read_raw_gdf(
    file_path,
    preload=True
)


# =====================
# 2. 提取事件
# =====================

events, event_dict = mne.events_from_annotations(raw)


event_id = {
    'left_hand': 7,
    'right_hand': 8,
    'feet': 9,
    'tongue': 10
}


# =====================
# 3. Epoch
# =====================

epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=0.5,
    tmax=4,
    baseline=None,
    preload=True
)


# =====================
# 4. 删除EOG
# =====================

epochs.drop_channels(
    [
        'EOG-left',
        'EOG-central',
        'EOG-right'
    ]
)


print("去EOG后:")
print(epochs.get_data().shape)



# =====================
# 5. 8-30Hz滤波
# =====================

epochs.filter(
    l_freq=8,
    h_freq=30
)



X = epochs.get_data()


print("----------------")

print("滤波后:")
print(X.shape)
# =====================
# 6. 获取数据
# =====================

X = epochs.get_data()
y = epochs.events[:, -1]

# =====================
# 7. 标准化
# =====================

mean = X.mean(axis=2, keepdims=True)

std = X.std(axis=2, keepdims=True)


X_norm = (X - mean) / std



print("----------------")

print("标准化后:")
print(X_norm.shape)


print("均值:")
print(X_norm.mean())


print("标准差:")
print(X_norm.std())
# =====================
# 8. 保存数据
# =====================

import os


save_path = "data/processed"

os.makedirs(
    save_path,
    exist_ok=True
)


np.save(
    os.path.join(save_path, "A01T_X.npy"),
    X_norm
)


np.save(
    os.path.join(save_path, "A01T_y.npy"),
    y
)


print("----------------")

print("数据保存完成")