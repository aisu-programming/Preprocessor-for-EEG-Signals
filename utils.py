##### Libraries #####
import dotenv
dotenv.load_dotenv(".env")
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from typing import Dict





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
    cm_image:plt.Axes = sn.heatmap(cm_df, annot=True)
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