import time
import torch
import optuna
import argparse
from baselines_pytorch import train



MODEL = ["EEGNet", "GRU", "LSTM", "ATCNet"][0]
DATASET = ["BcicIv2a", "PhysionetMI", "Ofner"][1]
FRAMEWORK = ["pt", "tf"][0]
STUDY_NAME = f"{MODEL}_{DATASET}_{FRAMEWORK}"



def get_default_args():
    args = argparse.ArgumentParser().parse_args()
    args.model = MODEL
    args.dataset = DATASET
    args.mixed_signal = False
    args.mixing_source = 2
    args.mixing_duplicate = 2
    args.epochs = 1000
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.num_workers = 2
    args.save_plot = True
    args.show_plot = False
    args.auto_hps = True
    return args

def set_args_save_dir(args):
    args.save_dir = time.strftime(f"histories_search/{STUDY_NAME}/")
    args.save_dir += f"bs={args.batch_size:03d}"
    args.save_dir += f"_lr={args.learning_rate:.4f}"
    args.save_dir += f"_ld={args.lr_decay:.6f}"
    if MODEL in ["GRU", "LSTM"]:
        args.save_dir += f"_hc={args.hid_channels:03d}"
    elif MODEL in ["ATCNet"]:
        args.save_dir += f"_nw={args.num_windows}"
    args.save_dir += f"_os"
    # if args.original_signal: args.save_dir += f"_os"
    if args.mixed_signal: args.save_dir += f"_m{args.mixing_source}x{args.mixing_duplicate}"
    return args

def objective(trial):

    args = get_default_args()

    if MODEL in ["GRU", "LSTM"]:
        args.hid_channels = trial.suggest_categorical("hid_channels", [16, 32, 64, 128, 256])
    elif MODEL in ["ATCNet"]:
        # args.num_windows = trial.suggest_categorical("num_windows", [2, 3, 4])
        args.num_windows = 3

    args.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])  # , 128])
    args.learning_rate = trial.suggest_float("learning_rate", 0.0005, 0.05, log=True)
    args.lr_decay = trial.suggest_float("lr_decay", 0.99987, 0.99994)
    args = set_args_save_dir(args)

    _, _, valid_accs, valid_losses = train(args)
    return max(valid_accs)
    # return max(valid_accs), min(valid_losses)



if __name__ == "__main__":
    # study = optuna.create_study(directions=["maximize", "minimize"],
    study = optuna.create_study(directions=["maximize"],
                                study_name=STUDY_NAME,
                                storage=f"sqlite:///{STUDY_NAME}.db",
                                load_if_exists=True)
    study.optimize(objective, n_trials=100)
    print("Number of finished trials: ", len(study.trials))
    for i, best_trial in enumerate(study.best_trials):
        print(f"The {i}-th Pareto solution was found at Trial#{best_trial.number}.")
        print(f"  Params: {best_trial.params}")
        valid_acc, valid_loss = best_trial.values
        print(f"  Values: val_acc={valid_acc*100:.2f}%, val_loss={valid_loss:.5f}")