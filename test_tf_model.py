import warnings
import numpy as np
import models_tensorflow.EEGModels
from tensorflow.keras import metrics
from libs.dataset import BcicIv2aDataset
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)



dataset = BcicIv2aDataset()
inputs, truths = dataset.all_data_and_label
inputs = np.expand_dims(inputs, axis=-1)
print(inputs.shape, truths.shape)

model = models_tensorflow.EEGModels.EEGNet(
            nb_classes=4, Chans=inputs.shape[1], Samples=inputs.shape[2],
            dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
            dropoutType="Dropout")
model.load_weights("histories/04.24-00.27.41_70.11%_tf_EEGNet_BCIC-IV-2a_bs=32_lr=0.025_ld=0.99991/best_val_acc.h5")
preds = model(inputs)
print(preds.shape)

metric = metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None)
print(f"Accuracy: {metric(preds, truths).numpy()*100:.2f}%")