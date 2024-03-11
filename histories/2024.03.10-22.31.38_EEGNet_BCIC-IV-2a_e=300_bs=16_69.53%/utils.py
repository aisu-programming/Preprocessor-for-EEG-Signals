##### Libraries #####
import dotenv
dotenv.load_dotenv(".env")
import os
import mne
import scipy.io
import itertools
import numpy as np
from tqdm import tqdm
from typing import Literal, List, Dict





##### Classes #####
class Dataset():
    def __init__(
            self,
            name: Literal["BCIC-IV-2a"],
            subjects: List[int],
            l_freq: int = -np.Inf,
            h_freq: int =  np.Inf,
            *args,
            **kwargs,
        ) -> None:
        
        assert name in ["BCIC-IV-2a"], NotImplementedError("Unknown dataset option.")
        if h_freq != -np.Inf:
            assert l_freq >= 0, "Parameters 'l_freq' should be >= 0."
        assert h_freq >= 0, "Parameters 'l_freq' should be >= 0."
        assert l_freq < h_freq, "Parameters 'l_freq' must be lesser than h_freq."

        if name == "BCIC-IV-2a":

            if h_freq != np.Inf:
                assert h_freq <= 125, "Parameters 'l_freq' should be <= 125."
            import warnings
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            self.labels : Dict[str, np.ndarray] = {}
            self.data   : Dict[str, np.ndarray] = {}
            pbar = tqdm(itertools.product(subjects, ['T', 'E']), total=len(subjects)*2)
            for sub_id, te in pbar:
                pbar.set_description(f"Loading BCIC IV 2a dataset - A0{sub_id}{te}")

                gdf_path = os.environ["DATASET_DIR"] + f"\BCI Competition IV 2a\A0{sub_id}{te}.gdf"
                raw: mne.io.edf.edf.RawGDF = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=0)

                if l_freq == -np.Inf: l_freq = raw.info["highpass"]
                if h_freq ==  np.Inf: h_freq = raw.info["lowpass"]

                ### !!! Need to Check !!! ###

                # raw.filter(l_freq, h_freq, fir_design="firwin")
                raw.filter(l_freq, h_freq, method="iir", iir_params=dict(order=3, ftype="butter"), phase="zero", verbose=0)
                # , l_trans_bandwidth=l_freq, h_trans_bandwidth=10)
                raw = raw.resample(128)

                if kwargs["remove_EOG"]:
                    raw.info["bads"] += ["EOG-left", "EOG-central", "EOG-right"]
                    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads")
                else:
                    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True, stim=False, exclude="bads")

                events, _ = mne.events_from_annotations(raw, verbose=0)

                if te == 'T':
                    # left_hand = 769, right_hand = 770, foot = 771, tongue = 772
                    if sub_id == 4:
                        event_id = {"769": 5, "770": 6, "771": 7, "772": 8}
                    else:
                        event_id = {"769": 7, "770": 8, "771": 9, "772": 10}
                else:
                    event_id = {"769": 7}
                epochs = mne.Epochs(raw, events, event_id, tmin=0.5, tmax=2.5, proj=True,
                                    picks=picks, baseline=None, preload=True, verbose=0,
                                    on_missing="warn")  # "ignore"

                self.data[f"A0{sub_id}{te}"] = epochs.get_data()
                if te == 'T':
                    if sub_id == 4:
                        self.labels[f"A0{sub_id}T"] = epochs.events[:, -1] - 5
                    else:
                        self.labels[f"A0{sub_id}T"] = epochs.events[:, -1] - 7
                else:
                    mat_path = os.environ["DATASET_DIR"] + f"\BCI Competition IV 2a\A0{sub_id}E.mat"
                    self.labels[f"A0{sub_id}E"] = scipy.io.loadmat(mat_path)["classlabel"].flatten() - 1



class BcicIv2aDataset(Dataset):
    def __init__(
            self,
            subjects: List[int] = [ 1, 2, 3, 4, 5, 6, 7, 8, 9 ],
            l_freq: int = 0,
            h_freq: int = 38,
            remove_EOG: bool = True,
            *args, **kwargs
        ) -> None:
        kwargs["remove_EOG"] = remove_EOG
        super().__init__("BCIC-IV-2a", subjects, l_freq, h_freq, *args, **kwargs)





##### Test #####
if __name__ == "__main__":
    # dataset = BcicIv2aDataset(l_freq=0, h_freq=38)
    dataset = BcicIv2aDataset(subjects=[4], l_freq=0, h_freq=38)
    # print([ dataset.labels["A01T"].tolist().count(i) for i in range(1, 4+1) ])
    # print([ dataset.labels["A01E"].tolist().count(i) for i in range(1, 4+1) ])