##### Libraries #####
import dotenv
dotenv.load_dotenv(".env")
import os
import mne
import itertools
import numpy as np
from tqdm import tqdm
from typing import List
from pathlib import Path
from .base import BaseDataset





##### Classes #####
class InnerSpeechDataset(BaseDataset):
    def __init__(
            self,
            subject_id_list: List[int] = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
            l_freq: int = 0,
            h_freq: int = 38,
            fs: int = 128,
            t_start: float = 1.0,
            t_end  : float = 3.5,
            # 0 = "PRONOUNCED", 1 = "INNER", 2 = "VISUALIZED"
            condition_list: List[int] = [ 1 ],
            # 0 = "UP", 1 = "DOWN", 2 = "RIGHT", 3 = "LEFT"
            class_list: List[int] = [ 0, 1, 2, 3 ],
        ) -> None:
        super().__init__()

        condition_class_list = []
        for cnd in condition_list:
            condition_class_list.extend([ cls+4*cnd for cls in class_list])

        self.class_number = len(condition_class_list)

        root_dir = Path(os.environ["DATASET_DIR"]) / "Inner Speech" / "derivatives"
        pbar = tqdm(list(itertools.product(subject_id_list, [1, 2, 3])))
        for sub_id, sess_id in pbar:

            if sub_id not in self.data:
                self.data[sub_id] = {}
                self.labels[sub_id] = {}

            str_sub, str_ses = f"sub-{sub_id:02}", f"ses-{sess_id:02}"
            pbar.set_description(f"Loading Inner Speech dataset - {str_sub}_{str_ses}")

            data = mne.read_epochs(
                root_dir / str_sub / str_ses / f"{str_sub}_{str_ses}_eeg-epo.fif",
                verbose="WARNING")
            
            if l_freq == -np.Inf: l_freq = data.info["highpass"]
            if h_freq ==  np.Inf: h_freq = data.info["lowpass"]


            ### !!! Need to Check !!! ###
            # data.filter(l_freq, h_freq, fir_design="firwin")
            data.filter(l_freq, h_freq, method="iir", iir_params=dict(order=3, ftype="butter"), phase="zero", verbose=0)
            # , l_trans_bandwidth=l_freq, h_trans_bandwidth=10)
            data = data.resample(fs)


            data = data._data
            start = max(round(t_start*fs), 0)
            end   = min(round(t_end*fs), data.shape[2])
            data = data[:, :, start:end]
            label = np.load(
                root_dir / str_sub / str_ses / f"{str_sub}_{str_ses}_events.dat",
                allow_pickle=True).squeeze()
            
            filtered_data, filtered_label = [], []
            for condition, class_ in itertools.product(condition_list, class_list):
                data_f = data[label[:, 2] == condition]
                label_f = label[label[:, 2] == condition]
                data_f = data_f[label_f[:, 1] == class_]
                label_f = label[label[:, 2] == condition]
                label_f = np.array([[ cls==(condition*4+class_) for cls in condition_class_list ]
                                    for _ in range(data_f.shape[0]) ])
                filtered_data.append(data_f)
                filtered_label.append(label_f)

            self.data[sub_id][sess_id] = np.vstack(filtered_data)
            self.labels[sub_id][sess_id] = np.vstack(filtered_label)