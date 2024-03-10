##### Libraries #####
import dotenv
dotenv.load_dotenv(".env")
import os
import mne
import numpy as np
from typing import Literal





##### Classes #####
class Dataset():
    def __init__(
            self,
            name: Literal["BCIC_IV_2a"],
            l_freq: int = -np.Inf,
            h_freq: int =  np.Inf,
            *args,
            **kwargs,
        ) -> None:
        
        assert name in ["BCIC_IV_2a"], NotImplementedError("Unknown dataset option.")
        if h_freq != -np.Inf:
            assert l_freq >= 0, "Parameters 'l_freq' should be >= 0."
        assert h_freq >= 0, "Parameters 'l_freq' should be >= 0."
        assert l_freq < h_freq, "Parameters 'l_freq' must be lesser than h_freq."

        if name == "BCIC_IV_2a":
            if h_freq != np.Inf:
                assert h_freq <= 125, "Parameters 'l_freq' should be <= 125."

            gdf_path = os.environ["DATASET_DIR"] + "\BCI Competition IV 2a\A01T.gdf"
            raw: mne.io.edf.edf.RawGDF = mne.io.read_raw_gdf(gdf_path)
            # print("raw.info:", raw.info)
            # print("\n\nraw.ch_names:", raw.ch_names)

            events, _ = mne.events_from_annotations(raw)
            raw.load_data()
            if l_freq == -np.Inf: l_freq = raw.info["highpass"]
            if h_freq ==  np.Inf: h_freq = raw.info["lowpass"]
            raw.filter(l_freq, h_freq, fir_design="firwin")         # !!! Need to Check

            if kwargs["remove_EOG"]:
                raw.info["bads"] += ["EOG-left", "EOG-central", "EOG-right"]
                picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads")
            else:
                picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True, stim=False, exclude="bads")

            # left_hand = 769, right_hand = 770, foot = 771, tongue = 772
            event_id = dict({"769": 7, "770": 8, "771": 9, "772": 10})
            epochs = mne.Epochs(raw, events, event_id,
                                tmin=1.0, tmax=4.0, proj=True,
                                picks=picks, baseline=None, preload=True)

            self.labels: np.ndarray = epochs.events[:, -1] - 7  # + 1
            self.data:   np.ndarray = epochs.get_data()



class BcicIv2aDataset(Dataset):
    def __init__(
            self,
            l_freq: int = 0,
            h_freq: int = 38,
            remove_EOG: bool = True,
            *args, **kwargs
        ) -> None:
        kwargs["remove_EOG"] = remove_EOG
        super().__init__("BCIC_IV_2a", l_freq, h_freq, *args, **kwargs)





##### Test #####
if __name__ == "__main__":
    dataset = BcicIv2aDataset(l_freq=0, h_freq=40)
    print(dataset.labels.shape, dataset.data.shape)