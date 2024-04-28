import random
random.seed(0)
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

    def splitted_data_and_label(
            self,
            ratio: float = 0.8
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_data, train_label, valid_data, valid_label = [], [], [], []
        for sub_id in self.data:
            for sess_id in self.data[sub_id]:
                self.data[sub_id][sess_id]
                self.labels[sub_id][sess_id]
                random_idx = list(range(len(self.data[sub_id][sess_id])))
                random.shuffle(random_idx)
                random_idx = np.array(random_idx)
                length = int(len(self.data[sub_id][sess_id])*ratio)
                train_data.append(self.data[sub_id][sess_id][random_idx[:length]])
                train_label.append(self.labels[sub_id][sess_id][random_idx[:length]])
                valid_data.append(self.data[sub_id][sess_id][random_idx[length:]])
                valid_label.append(self.labels[sub_id][sess_id][random_idx[length:]])
                # train_data.append(self.data[sub_id][sess_id][:length])
                # train_label.append(self.labels[sub_id][sess_id][:length])
                # valid_data.append(self.data[sub_id][sess_id][length:])
                # valid_label.append(self.labels[sub_id][sess_id][length:])
        return np.vstack(train_data), np.vstack(train_label), \
               np.vstack(valid_data), np.vstack(valid_label)