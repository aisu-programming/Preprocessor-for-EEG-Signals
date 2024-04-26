import time
import torch
import optuna
import argparse
from train_preprocessor_pt import train



DEFAULT_PREPROCESSOR = ["Transformer"][0]
DEFAULT_CLASSIFIER = ["EEGNet", "GRU", "LSTM", "ATCNet"][0]
DEFAULT_DATASET = ["BcicIv2a", "PhysionetMI", "Ofner"][0]
DEFAULT_FRAMEWORK = "pt"  # ["pt", "tf"][0]



def args_set_defaults(args):
    args.study_name = f"{args.preprocessor}_{args.classifier}_" + \
                      f"{args.dataset}_{args.framework}"
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
    args.save_dir = time.strftime(f"histories_pre_search/{args.study_name}/")
    args.save_dir += f"bs={args.batch_size:03d}"
    args.save_dir += f"_lr={args.learning_rate:.4f}"
    args.save_dir += f"_ld={args.lr_decay:.6f}"
    if args.preprocessor == "Transformer":
        args.save_dir += f"_nl={args.num_layers}_nh={args.num_heads}"
        args.save_dir += f"_fd={args.ffn_dim}_do={args.dropout}"
    return args


def objective(trial, args):

    if args.classifier == "EEGNet" and args.dataset == "BcicIv2a":
        args.classifier_weights = \
            "histories/04.24-17.54.06_70.45%_pt_EEGNet_BcicIv2a_bs=16_lr=0.03_ld=0.99993_os/best_valid_acc.pt"
    else:
        raise NotImplementedError

    if args.preprocessor == "Transformer":
        args.num_layers = trial.suggest_categorical("num_layers", [1, 2, 3, 4, 5])
        args.num_heads  = trial.suggest_categorical("num_heads", [3, 4, 6, 8])
        args.ffn_dim    = trial.suggest_categorical("ffn_dim", [128, 256, 512])
        args.dropout    = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3])

    args.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])  # , 128])
    args.learning_rate = trial.suggest_float("learning_rate", 0.0005, 0.05, log=True)
    args.lr_decay = trial.suggest_float("lr_decay", 0.99987, 0.99994)
    args = set_args_save_dir(args)
    return train(args)[2]   # valid acc


def main(args):
    args = args_set_defaults(args)
    study = optuna.create_study(directions=["maximize"],
                                study_name=args.study_name,
                                storage=f"sqlite:///{args.study_name}.db",
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
        "-p", "--preprocessor", type=str, default=DEFAULT_PREPROCESSOR,
        help="The preprocessor to be trained. Options: ['Transformer'].")
    parser.add_argument(
        "-c", "--classifier", type=str, default=DEFAULT_CLASSIFIER,
        help="The classifier to be used. " + \
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