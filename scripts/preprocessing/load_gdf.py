import mne


file_path = "D:\\EEG_BCI_Project\\data\\raw\\BCICIV_2a_gdf\\A01T.gdf"


raw = mne.io.read_raw_gdf(
    file_path,
    preload=True
)


print(raw)

print("----------------")

print(raw.ch_names)

print("----------------")

print(raw.info)