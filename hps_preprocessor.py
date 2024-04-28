import time
import torch
import optuna
import argparse
from train_preprocessor_pt import train



DEFAULT_PREPROCESSOR = ["LSTM", "Transformer"][0]
DEFAULT_CLASSIFIER = ["EEGNet", "GRU", "LSTM", "ATCNet"][0]
DEFAULT_DATASET = ["BcicIv2a", "PhysionetMI", "Ofner"][0]
DEFAULT_FRAMEWORK = "pt"  # ["pt", "tf"][0]



def args_set_defaults(args):
    args.study_name = f"{args.preprocessor}_{args.classifier}_" + \
                      f"{args.dataset}_{args.framework}"
    args.epochs = 1000
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.num_workers = 2
    args.save_plot = True
    args.show_plot = False
    args.auto_hps = True
    return args


def set_args_save_dir(args):
    args.save_dir = time.strftime(f"histories_pre_search/{args.study_name}/")
    args.save_dir += f"slf={args.sig_loss_factor:03d}"
    args.save_dir += f"_bs={args.batch_size:03d}"
    args.save_dir += f"_lr={args.learning_rate:.4f}"
    args.save_dir += f"_ld={args.lr_decay:.6f}"
    if args.preprocessor == "LSTM":
        args.save_dir += f"_nl={args.num_layers}_hs={args.hidden_size:03d}_do={args.dropout:.2f}"
    elif args.preprocessor == "Transformer":
        args.save_dir += f"_nl={args.num_layers}_nh={args.num_heads:02d}"
        args.save_dir += f"_fd={args.ffn_dim:03d}_do={args.dropout:.2f}"
    return args


def objective(trial, args):

    if args.classifier == "EEGNet" and args.dataset == "BcicIv2a":
        args.classifier_weights = \
            "histories_cls/EEGNet_BcicIv2a_pt/68.03%_bs=064_lr=0.0009_ld=0.999910_k1=32_k2=32_do=0.16/best_valid_acc.pt"
    elif args.classifier == "ATCNet" and args.dataset == "BcicIv2a":
        args.classifier_weights = \
            "histories_cls/ATCNet_BcicIv2a_pt/65.24%_bs=064_lr=0.0010_ld=0.999891_nw=3_cps=9/best_valid_acc.pt"
    else:
        raise NotImplementedError

    if args.preprocessor == "LSTM":
        args.num_layers  = trial.suggest_categorical("num_layers", [1, 2, 3])
        args.hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256])
        args.dropout     = trial.suggest_float("dropout", 0.0, 0.9)
    elif args.preprocessor == "Transformer":
        args.num_layers = trial.suggest_categorical("num_layers", [1, 2, 3, 4, 5])
        args.num_heads  = trial.suggest_categorical("num_heads", [3, 4, 6, 8])
        args.ffn_dim    = trial.suggest_categorical("ffn_dim", [128, 256, 512])
        args.dropout    = trial.suggest_float("dropout", 0.0, 0.9)
    else:
        raise NotImplementedError

    args.sig_loss_factor = trial.suggest_categorical("sig_loss_factor", [1, 2, 3, 5, 10, 20, 30, 50, 70, 100])
    args.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    args.learning_rate = trial.suggest_float("learning_rate", 0.0005, 0.05, log=True)
    args.lr_decay = trial.suggest_float("lr_decay", 0.99987, 0.99993)
    args = set_args_save_dir(args)
    return train(args)[2]   # valid acc


def main(args):
    args = args_set_defaults(args)
    study = optuna.create_study(directions=["maximize"],
                                study_name=args.study_name,
                                storage=f"sqlite:///hps_p_{args.study_name}.db",
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
        help="The preprocessor to be trained. Options: ['LSTM', 'Transformer'].")
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