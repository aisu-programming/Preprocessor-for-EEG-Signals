##### Libraries #####
import dotenv
dotenv.load_dotenv(".env")
import os
import mne
import itertools
import numpy as np
from typing import List
from pathlib import Path
from libs.Inner_Speech_Dataset.Python_Processing.Data_processing \
    import select_time_window, transform_for_classificator
from .base import BaseDataset





##### Classes #####
class InnerSpeechDataset(BaseDataset):
    def __init__(
            self,
            subject_id_list: List[int] = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
        ) -> None:
        super().__init__()
        for sub_id, sess_id in itertools.product(subject_id_list, [0, 1, 2]):
            data, label = extract_data_from_subject(
                Path(os.environ["DATASET_DIR"]) / "Inner Speech", sub_id, sess_id)
            self.data[sub_id][sess_id] = data
            self.labels[sub_id][sess_id] = label





##### Functions #####
def extract_data_from_subject(root_dir: str, sub_id: int, sess_id: int) -> tuple[np.ndarray, np.ndarray]:
    str_sub, str_ses = f"sub-{sub_id:02}", f"ses-{sess_id:02}"
    data = mne.read_epochs(
            Path(root_dir) / "derivatives" / str_sub / str_ses / f"{str_sub}_{str_ses}_eeg-epo.fif",
        verbose="WARNING")._data
    label = np.load(
            Path(root_dir) / "derivatives" / str_sub / str_ses / f"{str_sub}_{str_ses}_events.dat",
        allow_pickle=True).squeeze()
    return data, label