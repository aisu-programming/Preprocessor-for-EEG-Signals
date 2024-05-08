import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
from libs.dataset import BcicIv2aDataset
warnings.filterwarnings(action="ignore", category=FutureWarning)



DIR = "histories_pre/Transformer_EEGNet_BcicIv2a_pt/66.05%_slf=010_bs=064_lr=0.0016_ld=0.999911_nl=1_nh=06_fd=128_do=0.89"
preprocessor = torch.load(f"{DIR}/best_valid_acc.pt")

raw_signal = BcicIv2aDataset(subject_id_list=[1]).data[1][0][:32]
output: torch.Tensor = preprocessor(torch.tensor(raw_signal).cuda().float())
adj_signal: np.ndarray = output.detach().cpu().numpy()
print(adj_signal.shape)

plt.rcParams.update(plt.rcParamsDefault)
fig, axes = plt.subplots(7, 3, figsize=(15, 10))
fig.suptitle("BcicIv2aDataset: subject 1: trial 1: batch 1")
for i in range(7):
    for j in range(3):
        axes[i][j].plot(raw_signal[0, i*3+j], label="raw")
        axes[i][j].plot(adj_signal[0, i*3+j], label="adj")
        axes[i][j].set_ylabel(f"Channel: {i*3+j+1}")
        axes[i][j].legend()
        axes[i][j].grid()
plt.tight_layout()
plt.savefig(f"{DIR}/preprocessed_signal.png")