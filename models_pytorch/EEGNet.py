##### Libraries #####
import torch
import torch.nn as nn
from typing import Literal





##### Classes #####
class EEGNet(nn.Module):
    def __init__(
            self,
            nb_classes,
            Chans: int = 64,
            Samples: int = 128,
            dropoutRate: float = 0.5,
            kernLength: int = 64,
            F1: int = 8,
            D: int = 2,
            F2: int = 16,
            norm_rate: float = 0.25,
            dropoutType: Literal["Dropout", "SpatialDropout2D"] = "Dropout"
        ):
        super(EEGNet, self).__init__()

        assert dropoutType in ["Dropout", "SpatialDropout2D"], \
            "dropoutType must be one of SpatialDropout2D or Dropout, passed as a string."

        # Dropout Type
        if dropoutType == 'SpatialDropout2D':
            self.dropoutType = nn.Dropout2d(dropoutRate)
        elif dropoutType == 'Dropout':
            self.dropoutType = nn.Dropout(dropoutRate)
        else:
            raise ValueError("dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.")
        
        # Block 1
        self.block1_conv2d = nn.Conv2d(1, F1, (1, kernLength), padding="same", bias=False)
        self.block1_batchnorm = nn.BatchNorm2d(F1)
        self.block1_depthwise = nn.Conv2d(F1, F1*D, (Chans, 1), groups=F1, bias=False,
                                          padding=(0,0), padding_mode="zeros")
        self.block1_depthnorm = nn.BatchNorm2d(F1*D)
        self.block1_activation = nn.ELU()
        self.block1_avgpool = nn.AvgPool2d((1, 4))
        
        # Block 2
        self.block2_sepconv2d = nn.Conv2d(F1*D, F2, (1, 16), groups=F1*D, bias=False, padding="same")
        self.block2_conv2d = nn.Conv2d(F2, F2, kernel_size=1, bias=False)
        self.block2_batchnorm = nn.BatchNorm2d(F2)
        self.block2_activation = nn.ELU()
        self.block2_avgpool = nn.AvgPool2d((1, 8))
        
        # Output Layer
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * (Samples // 32), nb_classes)  # Adjust the size according to your pooling and stride
        self.softmax = nn.Softmax(dim=1)
        
        # Apply max norm constraint to certain layers
        self.block1_depthwise.weight.data = torch.renorm(self.block1_depthwise.weight.data, p=2, dim=0, maxnorm=1)
        self.dense.weight.data = torch.renorm(self.dense.weight.data, p=2, dim=0, maxnorm=norm_rate)

    def forward(self, x):
        # Block 1
        x = self.block1_conv2d(x)
        x = self.block1_batchnorm(x)
        x = self.block1_depthwise(x)
        x = self.block1_depthnorm(x)
        x = self.block1_activation(x)
        x = self.block1_avgpool(x)
        x = self.dropoutType(x)
        
        # Block 2
        x = self.block2_sepconv2d(x)
        x = self.block2_conv2d(x)
        x = self.block2_batchnorm(x)
        x = self.block2_activation(x)
        x = self.block2_avgpool(x)
        x = self.dropoutType(x)
        
        # Output Layer
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x