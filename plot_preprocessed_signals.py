import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
from libs.dataset import BcicIv2aDataset
warnings.filterwarnings(action="ignore", category=FutureWarning)



dir = "histories_pre/Transformer_EEGNet_BcicIv2a_pt/66.05%_slf=010_bs=064_lr=0.0016_ld=0.999911_nl=1_nh=06_fd=128_do=0.89"
preprocessor = torch.load(f"{dir}/best_valid_acc.pt")

raw_signal = BcicIv2aDataset(subject_id_list=[1]).data[1][0][:32]
output: torch.Tensor = preprocessor(torch.tensor(raw_signal).cuda().float())
adj_signal: np.ndarray = output.detach().cpu().numpy()
print(adj_signal.shape)

num_plot_channel = 4

plt.rcParams.update(plt.rcParamsDefault)
fig, axes = plt.subplots(num_plot_channel, 1, figsize=(12, 8))
fig.suptitle("BcicIv2aDataset: subject 1: trial 1: batch 1")
for i in range(num_plot_channel):
    axes[i].plot(raw_signal[0, i], label="raw")
    axes[i].plot(adj_signal[0, i], label="adj")
    axes[i].set_ylabel(f"Channel: {i+1}")
    axes[i].legend()
    axes[i].grid()
plt.tight_layout()
plt.savefig(f"{dir}/preprocessed_signal.png")