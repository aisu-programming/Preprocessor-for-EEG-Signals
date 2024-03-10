import mne
import matplotlib.pyplot as plt

gdf_path = "datasets\BCI Competition IV 2a\A01E.gdf"
raw: mne.io.edf.edf.RawGDF = mne.io.read_raw_gdf(gdf_path, preload=True)
# print(raw.ch_names)
# raw.load_data()

print(raw, raw.info)
fig = raw.plot()
fig.subplots_adjust(top=0.94, bottom=0.08, right=0.99)
plt.show()

# raw.filter(1, 40)

# picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads")
# ica = mne.preprocessing.ICA(n_components=25, random_state=97, max_iter=800)
# ica.fit(raw)
# ica.plot_components(picks)

# ica.exclude = [0, 1]
# ica.apply(raw)

# fig = raw.plot()
# fig.subplots_adjust(top=0.94, bottom=0.08, right=0.99)
# plt.show()