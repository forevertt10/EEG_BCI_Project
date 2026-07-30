import mne
import numpy as np


# =====================
# 1. 读取数据
# =====================

file_path = "data/raw/BCICIV_2a_gdf/A01T.gdf"


raw = mne.io.read_raw_gdf(
    file_path,
    preload=True
)


# =====================
# 2. 提取事件
# =====================

events, event_dict = mne.events_from_annotations(raw)


# =====================
# 3. 设置运动想象事件
# =====================

event_id = {
    'left_hand': 7,
    'right_hand': 8,
    'feet': 9,
    'tongue': 10
}


# =====================
# 4. 创建Epoch
# =====================

epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=0.5,
    tmax=4.0,
    baseline=None,
    preload=True
)


print(epochs)



# =====================
# 5. 查看数据维度
# =====================

X = epochs.get_data()

y = epochs.events[:, -1]


print("----------------")

print("X shape:")
print(X.shape)


print("y shape:")
print(y.shape)