##### Libraries #####
import dotenv
dotenv.load_dotenv(".env")
import os
import time
import utils
import shutil
import argparse
import numpy as np
np.random.seed(0)
import tensorflow as tf
tf.random.set_seed(0)
tf.keras.utils.set_random_seed(0)
import models_tensorflow.EEGModels
from typing import Literal
from tensorflow.keras import backend
from tensorflow.keras import utils as tf_utils
backend.set_image_data_format("channels_last")
from sklearn.model_selection import train_test_split





##### Classes #####
class RecordLearningRate(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        # print(f"\nCurrent learning rate: {lr:.10f}")
        logs = logs or {}
        logs["lr"] = lr





##### Functions #####
def backup_files(args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir)
    shutil.copy(__file__, args.save_dir)
    shutil.copy(utils.__file__, args.save_dir)
    shutil.copy(models_tensorflow.EEGModels.__file__, args.save_dir)
    with open(f"{args.save_dir}/args.txt", 'w') as record_txt:
        for key, value in args._get_kwargs():
            record_txt.write(f"{key}={value}\n")
    return



def train(
        model,
        training_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        epochs: int,
        batch_size: int,
        initial_learning_rate: float,
        decay_steps: int,
        decay_rate: float,
        save_dir: str,
        save_plot: bool = True,
        show_plot: bool = False,
    ) -> None:

    X_train, y_train, X_val, y_val = training_data
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs, verbose=1,
                        validation_data=(X_val, y_val),
                        callbacks=[ RecordLearningRate() ])
    utils.plot_history(history.history, "EEGNet",
                       f"{save_dir}/history_plot.png", save_plot, show_plot)

    best_acc_val = max(history.history["val_accuracy"]) * 100
    new_save_dir = save_dir.replace("histories_tmp/", '')
    new_save_dir = new_save_dir.split('_', 1)
    new_save_dir = f"histories_tmp/{new_save_dir[0]}_{best_acc_val:.2f}%_{new_save_dir[1]}"
    os.rename(save_dir, new_save_dir)
    return



def main(args: argparse.Namespace) -> None:

    assert args.model in ["EEGNet", "DeepConvNet"], \
        "Invalid value for parameter 'model'."
    assert args.dataset in ["BCIC-IV-2a"], \
        "Invalid value for parameter 'dataset'."

    if args.dataset == "BCIC-IV-2a":
        dataset = utils.BcicIv2aDataset()  # l_freq=4
        X_all = np.concatenate([ v for v in dataset.data.values()   ], axis=0)
        y_all = np.concatenate([ v for v in dataset.labels.values() ], axis=0)
        X_train, X_val, y_train, y_val = \
            train_test_split(X_all, y_all, test_size=0.2, random_state=0, shuffle=True)
        X_train = np.expand_dims(X_train, -1)
        X_val   = np.expand_dims(X_val,   -1)
        y_train = tf_utils.to_categorical(y_train)
        y_val   = tf_utils.to_categorical(y_val)
        training_data = (X_train, y_train, X_val, y_val)

    if args.model == "EEGNet":
        model = models_tensorflow.EEGModels.EEGNet(
                    nb_classes=4, Chans=X_train.shape[1], Samples=X_train.shape[2],
                    dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
                    dropoutType="Dropout")
    elif args.model == "DeepConvNet":
        model = models_tensorflow.EEGModels.DeepConvNet(
                    nb_classes=4, Chans=X_train.shape[1],
                    Samples=X_train.shape[2], dropoutRate=0.5)

    train(
        model=model,
        training_data=training_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        initial_learning_rate=args.initial_learning_rate,
        decay_steps=args.decay_steps,
        decay_rate=args.decay_rate,
        save_dir=args.save_dir,
        save_plot=args.save_plot,
        show_plot=args.show_plot,
    )





##### Execution #####
if __name__ == "__main__":

    default_save_dir_pre = time.strftime("histories_tmp/%m.%d-%H.%M.%S_tf")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="DeepConvNet",
        help="The model to be trained. Options: ['EEGNet', 'DeepConvNet']."
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="BCIC-IV-2a",
        help="The dataset used for training."
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=300,
        help="The total epochs (iterations) of training."
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=32,
        help="The batch size of training input."
    )
    parser.add_argument(
        "-ilr", "--initial_learning_rate", type=float, default=1e-3,
        help="The initial learning rate of the optimizer for training."
    )
    parser.add_argument(
        "-ds", "--decay_steps", type=int, default=1,
        help="The decay step of the optimizer for training."
    )
    parser.add_argument(
        "-dr", "--decay_rate", type=float, default=0.9998,
        help="The decay rate of the optimizer for training."
    )
    parser.add_argument(
        "-sd", "--save_dir", type=str, default=None,
        help="The path to save all history files."
    )
    parser.add_argument(
        "-s", "--save_plot", type=bool, default=True,
        help="Whether to save the training history plot."
    )
    parser.add_argument(
        "--show_plot", type=bool, default=True,
        help="Whether to show the training history plot."
    )
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir =  default_save_dir_pre
        args.save_dir += f"_{args.model}_{args.dataset}"
        args.save_dir += f"_bs={args.batch_size}"
        args.save_dir += f"_lr={args.initial_learning_rate}"
        args.save_dir += f"_ld={args.decay_rate}"

    backup_files(args)
    main(args)