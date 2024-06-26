##### Libraries #####
import dotenv
dotenv.load_dotenv(".env")
import os
import time
import utils
import shutil
import argparse
import warnings
import numpy as np
import tensorflow as tf
import models_tensorflow.EEGModels
from tensorflow.keras import backend
backend.set_image_data_format("channels_last")
from libs.dataset import BcicIv2aDataset, PhysionetMIDataset, Ofner2017Dataset  # , InnerSpeechDataset

np.random.seed(0)
tf.random.set_seed(0)
tf.keras.utils.set_random_seed(0)
warnings.filterwarnings(action="ignore", category=FutureWarning)




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
        save_model: bool = True,
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
    
    callbacks = [ RecordLearningRate() ]
    if save_model:
        callbacks.extend([
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{save_dir}/best_val_acc.h5",
                monitor="val_accuracy",
                mode="max",
                save_weights_only=True,
                save_best_only=True),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{save_dir}/best_val_loss.h5",
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True),
        ])
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs, verbose=1,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks)
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
    # assert args.dataset in ["BcicIv2a", "PhysionetMI", "Ofner", "InnerSpeech"], \
    assert args.dataset in ["BcicIv2a", "PhysionetMI", "Ofner"], \
        "Invalid value for parameter 'dataset'."

    if args.dataset == "BcicIv2a":
        dataset = BcicIv2aDataset()
    if args.dataset == "PhysionetMI":
        dataset = PhysionetMIDataset()
    elif args.dataset == "Ofner":
        dataset = Ofner2017Dataset()
    # elif args.dataset == "InnerSpeech":
    #     dataset = InnerSpeechDataset()

    train_inputs, train_truths, valid_inputs, valid_truths = \
        dataset.splitted_data_and_label()
    train_inputs = np.expand_dims(train_inputs, axis=-1)
    valid_inputs = np.expand_dims(valid_inputs, axis=-1)
    print(train_inputs.shape, train_truths.shape)
    print(valid_inputs.shape, valid_truths.shape)

    return

    if args.model == "EEGNet":
        model = models_tensorflow.EEGModels.EEGNet(
                    nb_classes=4, Chans=train_inputs.shape[1], Samples=train_inputs.shape[2],
                    dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
                    dropoutType="Dropout")
    elif args.model == "DeepConvNet":
        model = models_tensorflow.EEGModels.DeepConvNet(
                    nb_classes=4, Chans=train_inputs.shape[1],
                    Samples=train_inputs.shape[2], dropoutRate=0.5)

    train(
        model=model,
        training_data=(train_inputs, train_truths,
                       valid_inputs, valid_truths),
        epochs=args.epochs,
        batch_size=args.batch_size,
        initial_learning_rate=args.initial_learning_rate,
        decay_steps=args.decay_steps,
        decay_rate=args.decay_rate,
        save_dir=args.save_dir,
        save_model=args.save_model,
        save_plot=args.save_plot,
        show_plot=args.show_plot,
    )





##### Execution #####
if __name__ == "__main__":

    default_save_dir_pre = time.strftime("histories_tmp/%m.%d-%H.%M.%S_tf")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="EEGNet",
        help="The model to be trained. Options: ['EEGNet', 'DeepConvNet'].")
    parser.add_argument(
        "-d", "--dataset", type=str, default="Ofner",
        help="The dataset used for training. " + \
             "Options: ['BcicIv2a', 'PhysionetMI', 'Ofner'].")
            #  "Options: ['BcicIv2a', 'PhysionetMI', 'Ofner', 'InnerSpeech'].")
    parser.add_argument(
        "-e", "--epochs", type=int, default=800,
        help="The total epochs (iterations) of training.")
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=32,
        help="The batch size of training input.")
    parser.add_argument(
        "-ilr", "--initial_learning_rate", type=float, default=0.03,
        help="The initial learning rate of the optimizer for training.")
    parser.add_argument(
        "-ds", "--decay_steps", type=int, default=1,
        help="The decay step of the optimizer for training.")
    parser.add_argument(
        "-dr", "--decay_rate", type=float, default=0.99992,
        help="The decay rate of the optimizer for training.")
    parser.add_argument(
        "-sd", "--save_dir", type=str, default=None,
        help="The path to save all history files.")
    parser.add_argument(
        "-sm", "--save_model", type=bool, default=True,
        help="Whether to save the best model.")
    parser.add_argument(
        "-sp", "--save_plot", type=bool, default=True,
        help="Whether to save the training history plot.")
    parser.add_argument(
        "--show_plot", type=bool, default=True,
        help="Whether to show the training history plot.")
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir =  default_save_dir_pre
        args.save_dir += f"_{args.model}_{args.dataset}"
        args.save_dir += f"_bs={args.batch_size}"
        args.save_dir += f"_lr={args.initial_learning_rate}"
        args.save_dir += f"_ld={args.decay_rate}"

    backup_files(args)
    main(args)