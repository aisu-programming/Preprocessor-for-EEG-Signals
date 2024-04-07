import numpy as np
from typing import Dict, Tuple


##### Classes #####
class BaseDataset():
    def __init__(self) -> None:
        self.class_number: int = 0
        self.data        : Dict[str, Dict[str, np.ndarray]] = {}
        self.labels      : Dict[str, Dict[str, np.ndarray]] = {}

    @property
    def all_data_and_label(self) -> Tuple[np.ndarray, np.ndarray]:
        data, label = [], []
        for sub_id in self.data:
            for sess_id in self.data[sub_id]:
                data.append(self.data[sub_id][sess_id])
                label.append(self.labels[sub_id][sess_id])
        return np.vstack(data), np.vstack(label)