##### Libraries #####
import dotenv
dotenv.load_dotenv(".env")
import os
import time
import shutil
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, Literal
from utils import BcicIv2aDataset
from tensorflow.keras import utils
from tensorflow.keras import backend
backend.set_image_data_format("channels_last")
from arl_eegmodels.EEGModels import EEGNet
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
    shutil.copy("utils.py", args.save_dir)
    with open(f"{args.save_dir}/args.txt", 'w') as record_txt:
        for key, value in args._get_kwargs():
            record_txt.write(f"{key}={value}\n")


def plot_history(
        history: Dict[str, list],
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
    fig.suptitle(f"Baseline: EEGNet")
    axs[0].plot(list(range(1, len(train_acc)+1)), train_acc, label=f"best train acc={max(train_acc)*100:.2f}%")
    axs[0].plot(list(range(1, len(val_acc)+1)),   val_acc,   label=f"best valid acc={max(val_acc)*100:.2f}%")
    axs[0].set_title("Accuracies")
    axs[0].legend()
    axs[0].grid()
    axs[1].plot(list(range(1, len(train_loss)+1)), train_loss, label=f"best train loss={min(train_loss):.5f}")
    axs[1].plot(list(range(1, len(val_loss)+1)),   val_loss,   label=f"best valid loss={min(val_loss):.5f}")
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


def baseline_EEGNet(
        dataset: Literal["BCIC-IV-2a"],
        epochs: int,
        batch_size: int,
        initial_learning_rate: float,
        decay_steps: int,
        decay_rate: float,
        save_dir: str,
        save_plot: bool = True,
        show_plot: bool = False,
    ) -> None:

    assert dataset in ["BCIC-IV-2a"], "Invalid value for parameter 'dataset'."

    if dataset == "BCIC-IV-2a":
        dataset = BcicIv2aDataset(subjects=[1])  # l_freq=4

    X_all = np.concatenate([ v for v in dataset.data.values()   ], axis=0)
    y_all = np.concatenate([ v for v in dataset.labels.values() ], axis=0)
    # print(X_all.shape, y_all.shape)

    X_train, X_val, y_train, y_val = \
        train_test_split(X_all, y_all, test_size=0.2, random_state=1, shuffle=True)
    X_train = np.reshape(X_train, [*X_train.shape, 1])
    X_val   = np.reshape(X_val,   [*X_val.shape,   1])
    y_train = utils.to_categorical(y_train)
    y_val   = utils.to_categorical(y_val)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    model_8_2 = EEGNet(nb_classes=4, Chans=X_train.shape[1], Samples=X_train.shape[2],
                       dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16, dropoutType="Dropout")
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
    model_8_2.compile(loss="categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])
    history = model_8_2.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs, verbose=1,
                            validation_data=(X_val, y_val),
                            callbacks=[ RecordLearningRate() ])
    plot_history(history.history, f"{save_dir}/history_plot.png", save_plot, show_plot)

    best_acc_val = max(history.history["val_accuracy"]) * 100
    os.rename(save_dir, f"{save_dir}_{best_acc_val:.2f}%")





##### Execution #####
if __name__ == "__main__":

    default_save_dir_pre = time.strftime("histories/%m.%d-%H.%M.%S")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="EEGNet",
        help="The model to be trained. Options: ['EEGNet']."
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="BCIC-IV-2a",
        help="The dataset used for training."
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=350,
        help="The total epochs (iterations) of training."
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=32,
        help="The batch size of training input."
    )
    parser.add_argument(
        "-ilr", "--initial_learning_rate", type=float, default=8e-4,
        help="The initial learning rate of the optimizer for training."
    )
    parser.add_argument(
        "-ds", "--decay_steps", type=int, default=1000,
        help="The decay step of the optimizer for training."
    )
    parser.add_argument(
        "-dr", "--decay_rate", type=float, default=0.9,
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
        args.save_dir += f"_ilr={args.initial_learning_rate}"
        args.save_dir += f"_ds={args.decay_steps}"

    backup_files(args)

    if args.model == "EEGNet":
        baseline_EEGNet(
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            initial_learning_rate=args.initial_learning_rate,
            decay_steps=args.decay_steps,
            decay_rate=args.decay_rate,
            save_dir=args.save_dir,
            save_plot=args.save_plot,
            show_plot=args.show_plot,
        )