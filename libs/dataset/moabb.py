##### Libraries #####
import dotenv
dotenv.load_dotenv(".env")
import os
import mne
import moabb
import urllib3
import warnings
import numpy as np
from tqdm import tqdm
from typing import Literal, List, Union
from moabb.datasets import PhysionetMI, Ofner2017
from moabb.paradigms import MotorImagery
from .base import BaseDataset

mne.set_config("MNE_DATA", os.environ["DATASET_DIR"])
mne.set_config("MNE_DATASETS_EEGBCI_PATH", os.environ["DATASET_DIR"])
moabb.set_log_level("error")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ModuleNotFoundError)





##### Classes #####
class MoabbDataset(BaseDataset):
    def __init__(
            self,
            dataset: Literal["PhysionetMI", "Ofner2017"],
            subject_id_list: List[int],
            l_freq : int,
            h_freq : int,
            fs     : int,
            t_start: float,
            t_end  : Union[float, None],
            condition_list: List[int],
            class_list: List[int],
        ) -> None:
        super().__init__()

        if condition_list != [1]:
            raise NotImplementedError("Execution option isn't implemented.")

        if dataset == "PhysionetMI":
            dataset = PhysionetMI()
            class_to_event = { 0: "left_hand", 1: "right_hand", 2: "hands", 3: "feet" }
            events = [ class_to_event[cid] for cid in class_list ]
        elif dataset == "Ofner2017":
            dataset = Ofner2017()
            class_to_event = { 0: "left_hand", 1: "right_hand",
                               2: "hands", 3: "feet",
                               4: "", 5: "" }
            events = [ class_to_event[cid] for cid in class_list ]

        self.class_number = len(events)
        paradigm = MotorImagery(events=events, n_classes=len(events),
                                fmin=l_freq, fmax=h_freq,
                                tmin=t_start, tmax=t_end, resample=fs)

        pbar = tqdm(subject_id_list)
        for sub_id in pbar:
            pbar.set_description(f"Loading {dataset} dataset - {sub_id}")
            if sub_id not in self.data:
                self.data[sub_id] = {0: []}
                self.labels[sub_id] = {0: []}
            data, label, _ = paradigm.get_data(dataset=dataset, subjects=[sub_id])
            self.data[sub_id][0] = data
            self.labels[sub_id][0] = np.array([ [ lbl==eve for eve in events ] for lbl in label ])



class PhysionetMIDataset(MoabbDataset):
    def __init__(
            self,
            subject_id_list: List[int] = list(range(1, 109+1)),
            l_freq : int = 0,
            h_freq : int = 38,
            fs     : int = 128,
            t_start: float = 0.0,
            t_end  : Union[float, None] = None,
            # 0 = "Execution", 1 = "Imagery"
            condition_list: List[int] = [ 1 ],
            # 0 = "left_hand", 1 = "right_hand", 2 = "hands", 3 = "feet"
            class_list: List[int] = [ 0, 1, 2, 3 ],
        ) -> None:
        super().__init__("PhysionetMI",
                         subject_id_list,
                         l_freq,
                         h_freq,
                         fs,
                         t_start,
                         t_end,
                         condition_list,
                         class_list)
        """
        - Special trial number
          Most of the subject has 90 trials, except:
          - 72: Subject 100
          - 88: Subject 104
          - 114: Subject 88 and 92
        """
        return



class Ofner2017Dataset(MoabbDataset):
    def __init__(
            self,
            subject_id_list: List[int] = list(range(1, 15+1)),
            l_freq : int = 0,
            h_freq : int = 38,
            fs     : int = 128,
            t_start: float = 0.0,
            t_end  : Union[float, None] = None,
            # 0 = "Execution", 1 = "Imagery"
            condition_list: List[int] = [ 1 ],
            # elbow flexion/extension,
            # forearm supination/pronation
            # hand open/close
            # all with the right upper limb
            class_list: List[int] = [ 0, 1, 2, 3, 4, 5 ],
        ) -> None:
        super().__init__("Ofner2017",
                         subject_id_list,
                         l_freq,
                         h_freq,
                         fs,
                         t_start,
                         t_end,
                         condition_list,
                         class_list)
        return