##### Libraries #####
import dotenv
dotenv.load_dotenv(".env")
import itertools
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict
from matplotlib.axes import Axes
from libs.dataset.base import BaseDataset





##### Classes #####
class Metric():
    def __init__(self, length: int) -> None:
        self.length = length
        self.values = []

    def append(self, value) -> None:
        self.values.append(value)
        if len(self.values) > self.length: self.values.pop(0)
        return

    @ property
    def avg(self) -> float:
        return np.average(self.values)
    




##### Functions #####
def plot_confusion_matrix(
        cm_length: int,
        cm: np.ndarray,
        filename: str,
        title: str,
    ) -> None:
    cm_df = pd.DataFrame(cm, index=list(range(cm_length)), columns=list(range(cm_length)))
    plt.figure(figsize=(6, 5))
    cm_image: plt.Axes = sn.heatmap(cm_df, annot=True, fmt=".0f")
    cm_image.set_xlabel("prediction", fontsize=10)
    cm_image.set_ylabel("truth", fontsize=10)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return


def plot_history(
        history: Dict[str, list],
        model: str,
        output_path: str,
        save: bool = True,
        show: bool = False,
    ) -> None:

    train_acc  = history["accuracy"]
    val_acc    = history["val_accuracy"]
    train_loss = history["loss"]
    val_loss   = history["val_loss"]
    lr         = history["lr"]

    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    fig.suptitle(f"Baseline: {model}")
    axs[0].plot(list(range(1, len(train_acc)+1)), train_acc,
                label=f"best train acc={max(train_acc)*100:.2f}% @ {np.argmax(train_acc)+1}")
    axs[0].plot(list(range(1, len(val_acc)+1)), val_acc,
                label=f"best valid acc={max(val_acc)*100:.2f}% @ {np.argmax(val_acc)+1}")
    axs[0].set_title("Accuracies")
    axs[0].legend()
    axs[0].grid()
    axs[1].plot(list(range(1, len(train_loss)+1)), train_loss,
                label=f"best train loss={min(train_loss):.5f} @ {np.argmin(train_loss)+1}")
    axs[1].plot(list(range(1, len(val_loss)+1)), val_loss,
                label=f"best valid loss={min(val_loss):.5f} @ {np.argmin(val_loss)+1}")
    axs[1].set_title("Losses")
    axs[1].set_yscale("log")
    axs[1].legend()
    axs[1].grid()
    axs[2].plot(list(range(1, len(lr)+1)), lr)
    axs[2].set_title("Learning Rate")
    axs[2].set_yscale("log")
    axs[2].grid()

    plt.tight_layout()
    if show: plt.show()
    if save: fig.savefig(output_path)
    return


def plot_dataset(dataset: BaseDataset) -> None:
    
    sub_sess_list = list(itertools.product(dataset.data.keys(), range(dataset.class_number)))
    channel_number = dataset.data[1][1].shape[1]

    _, axes = plt.subplots(channel_number, len(sub_sess_list), figsize=(30, 15))
    pbar = tqdm(enumerate(sub_sess_list), total=len(sub_sess_list))
    for col, (sub_id, cls) in pbar:
        pbar.set_description(f"Processing sub-{sub_id+1}_cls-{cls+1}")
        for row, chn in enumerate(range(channel_number)):
            ax: Axes = axes[row][col]
            if row == 0: ax.set_title(f"{sub_id+1:02}:{cls+1}")
            if col == 0: ax.set_ylabel(f"{chn+1:02}", rotation=0, labelpad=10)
            for sess_id in dataset.data[sub_id]:
                data  = dataset.data[sub_id][sess_id]
                label = np.argmax(dataset.labels[sub_id][sess_id], axis=1)
                data = np.average(data[label==cls, chn, :], axis=0)
                ax.plot(data, label=sess_id+1, lw=0.6, alpha=0.6)
                ax.set_xticks([])
                ax.set_yticks([])
    print("Layout tighting... ", end='', flush=True)
    plt.tight_layout()
    print("Done.\nSaving plot... ", end='', flush=True)
    plt.savefig("test.png", dpi=300)
    print("Done.", flush=True)
    return





if __name__ == "__main__":
    from libs.dataset import BcicIv2aDataset, InnerSpeechDataset
    plot_dataset(BcicIv2aDataset())