##### Libraries #####
import dotenv
dotenv.load_dotenv(".env")
import os
import mne
import scipy.io
import itertools
import numpy as np
from tqdm import tqdm
from typing import List
from pathlib import Path
from .base import BaseDataset





##### Classes #####
class BcicIv2aDataset(BaseDataset):
    def __init__(
            self,
            subject_id_list: List[int] = [ 1, 2, 3, 4, 5, 6, 7, 8, 9 ],
            l_freq: int = 0,
            h_freq: int = 38,
            remove_EOG: bool = True,
        ) -> None:
        super().__init__()
        
        if h_freq != np.Inf:
            assert h_freq <= 125, "Parameters 'l_freq' should be <= 125."
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        pbar = tqdm(itertools.product(subject_id_list, ['T', 'E']),
                    total=len(subject_id_list)*2)
        for sub_id, te in pbar:
            pbar.set_description(f"Loading BCIC IV 2a dataset - A0{sub_id}{te}")

            sess_id = 0 if te == 'T' else 1
            if sub_id not in self.data:
                self.data[sub_id] = {}
                self.labels[sub_id] = {}

            gdf_path = Path(os.environ["DATASET_DIR"]) / "BCI Competition IV 2a" / f"A0{sub_id}{te}.gdf"
            raw: mne.io.edf.edf.RawGDF = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=0)

            if l_freq == -np.Inf: l_freq = raw.info["highpass"]
            if h_freq ==  np.Inf: h_freq = raw.info["lowpass"]

            ### !!! Need to Check !!! ###
            # raw.filter(l_freq, h_freq, fir_design="firwin")
            raw.filter(l_freq, h_freq, method="iir", iir_params=dict(order=3, ftype="butter"), phase="zero", verbose=0)
            # , l_trans_bandwidth=l_freq, h_trans_bandwidth=10)
            raw = raw.resample(128)

            if remove_EOG:
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

            try: 
                self.data[sub_id][sess_id] = epochs.get_data(copy=False) # set explicitly due to warning
            except TypeError:
                self.data[sub_id][sess_id] = epochs.get_data()

            if te == 'T':
                if sub_id == 4:
                    self.labels[sub_id][sess_id] = np.array([
                        [ lbl==cls for cls in range(4) ]
                            for lbl in epochs.events[:, -1] - 5 ])
                else:
                    self.labels[sub_id][sess_id] = np.array([
                        [ lbl==cls for cls in range(4) ]
                            for lbl in epochs.events[:, -1] - 7 ])
            else:
                mat_path = Path(os.environ["DATASET_DIR"]) / "BCI Competition IV 2a" / f"A0{sub_id}E.mat"
                self.labels[sub_id][sess_id] = np.array([
                        [ lbl==cls for cls in range(4) ] for lbl in
                            scipy.io.loadmat(str(mat_path))["classlabel"].flatten() - 1 ])