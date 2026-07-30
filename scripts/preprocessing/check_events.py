import mne


file_path = "data/raw/BCICIV_2a_gdf/A01T.gdf"


raw = mne.io.read_raw_gdf(
    file_path,
    preload=False
)


events, event_dict = mne.events_from_annotations(raw)


print("事件字典:")
print(event_dict)


print("----------------")


print("事件数量:")
print(len(events))


print("----------------")


print("前10个事件:")
print(events[:10])

import numpy as np


labels = events[:,2]


for event_id in [7,8,9,10]:

    count = np.sum(labels == event_id)

    print(
        event_id,
        "数量:",
        count
    )