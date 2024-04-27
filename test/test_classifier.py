import torch
import itertools
from torcheeg.models import EEGNet, GRU, LSTM, ATCNet



def main():
    inputs = torch.rand((32, 1, 22, 381))
    for k1, k2 in itertools.product([32, 64], [16, 32]):
        model = EEGNet(
            kernel_1=k1, kernel_2=k2,
            chunk_size=inputs.shape[3],
            num_electrodes=inputs.shape[2])
        assert model(inputs).shape == (32, 2)

    inputs = torch.rand((32, 22, 381))
    for hc in [16, 32, 64, 128, 256]:
        model = GRU(
            hid_channels=hc,
            num_electrodes=inputs.shape[1])
        assert model(inputs).shape == (32, 2)

    inputs = torch.rand((32, 22, 381))
    for hc in [16, 32, 64, 128, 256]:
        model = LSTM(
            hid_channels=hc,
            num_electrodes=inputs.shape[1])
        assert model(inputs).shape == (32, 2)

    inputs = torch.rand((32, 1, 22, 381))
    for nw, cps in itertools.product([2, 3, 4], [5, 7, 9]):
        model = ATCNet(
            num_windows=nw,
            conv_pool_size=cps,
            chunk_size=inputs.shape[3],
            num_electrodes=inputs.shape[2])
        assert model(inputs).shape == (32, 4)
    

    
if __name__ == "__main__":
    main()