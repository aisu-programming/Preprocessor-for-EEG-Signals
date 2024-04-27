import time
import torch
import optuna
import argparse
from train_classifier_pt import train



DEFAULT_MODEL = ["EEGNet", "GRU", "LSTM", "ATCNet"][0]
DEFAULT_DATASET = ["BcicIv2a", "PhysionetMI", "Ofner"][1]
DEFAULT_FRAMEWORK = ["pt", "tf"][0]



def args_set_defaults(args):
    args.study_name = f"{args.model}_{args.dataset}_{args.framework}"
    args.epochs = 1000
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.num_workers = 2
    args.save_plot = True
    args.show_plot = False
    args.auto_hps = True
    return args



def set_args_save_dir(args):
    args.save_dir = time.strftime(f"histories_cls_search/{args.study_name}/")
    args.save_dir += f"bs={args.batch_size:03d}"
    args.save_dir += f"_lr={args.learning_rate:.4f}"
    args.save_dir += f"_ld={args.lr_decay:.6f}"
    if args.model in ["EEGNet"]:
        args.save_dir += f"_k1={args.kernel_1}_k2={args.kernel_2}"
        args.save_dir += f"_do={args.dropout:.02f}"
    elif args.model in ["GRU", "LSTM"]:
        args.save_dir += f"_hc={args.hid_channels:03d}"
    elif args.model in ["ATCNet"]:
        args.save_dir += f"_nw={args.num_windows}"
    return args



def objective(trial, args):

    if args.model == "EEGNet":
        args.kernel_1 = trial.suggest_categorical("kernel_1", [32, 64])
        args.kernel_2 = trial.suggest_categorical("kernel_2", [16, 32])
        args.dropout  = trial.suggest_float("dropout", 0.1, 0.5)
        args.F1 = 8
        args.F2 = 16
        args.D = 2
    elif args.model in ["GRU", "LSTM"]:
        args.hid_channels = trial.suggest_categorical("hid_channels", [16, 32, 64, 128, 256])
    elif args.model in ["ATCNet"]:
        args.num_windows = trial.suggest_categorical("num_windows", [2, 3, 4])
        args.conv_pool_size = trial.suggest_categorical("conv_pool_size", [5, 7, 9])
        args.F1 = 16
        args.D = 2
        args.tcn_kernel_size = 4
        args.tcn_depth = 2

    args.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])  # , 128])
    args.learning_rate = trial.suggest_float("learning_rate", 0.0005, 0.05, log=True)
    args.lr_decay = trial.suggest_float("lr_decay", 0.99985, 0.99995)
    args = set_args_save_dir(args)
    return train(args)[2]



def main(args):
    args = args_set_defaults(args)
    study = optuna.create_study(directions=["maximize"],
                                study_name=args.study_name,
                                storage=f"sqlite:///hps_c_{args.study_name}.db",
                                load_if_exists=True)
    study.optimize(lambda trial: objective(trial, args), n_trials=100)

    print("Number of finished trials: ", len(study.trials))
    for i, best_trial in enumerate(study.best_trials):
        print(f"The {i}-th Pareto solution was found at Trial#{best_trial.number}.")
        print(f"  Params: {best_trial.params}")
        valid_acc, valid_loss = best_trial.values
        print(f"  Values: val_acc={valid_acc*100:.2f}%, val_loss={valid_loss:.5f}")
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default=DEFAULT_MODEL,
        help="The model to be trained. " + \
             "Options: ['EEGNet', 'GRU', 'LSTM', 'ATCNet'].")
    parser.add_argument(
        "-d", "--dataset", type=str, default=DEFAULT_DATASET,
        help="The dataset used for training. " + \
             "Options: ['BcicIv2a', 'PhysionetMI', 'Ofner'].")
    parser.add_argument(
        "-f", "--framework", type=str, default=DEFAULT_FRAMEWORK,
        help="Options: ['pt', 'tf'].")
    args = parser.parse_args()
    main(args)