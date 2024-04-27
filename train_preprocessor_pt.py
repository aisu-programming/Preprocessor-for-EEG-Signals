##### Libraries #####
import os
import time
import torch
import shutil
import random
import argparse
import warnings
import numpy as np
import torch.utils.data
from tqdm import tqdm
from typing import Union, Tuple, Literal
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
warnings.filterwarnings(action="ignore", category=FutureWarning)

import utils
from utils import Metric, plot_confusion_matrix, plot_history_sca
from libs.dataset import BcicIv2aDataset, PhysionetMIDataset, Ofner2017Dataset
from models_pytorch.preprocessor import PreLSTM, PreTransformer





##### Classes #####
class MyMapDataset(Dataset):
    def __init__(self,
                 inputs: Union[list, np.ndarray],
                 truths: Union[list, np.ndarray]) -> None:
        self.inputs = torch.Tensor(inputs)
        self.truths = torch.Tensor(truths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.truths[index]

    def __len__(self) -> int:
        return len(self.inputs)





##### Functions #####
def backup_files(args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir)
    shutil.copy(__file__, args.save_dir)
    shutil.copy(utils.__file__, args.save_dir)
    shutil.copytree("libs", f"{args.save_dir}/libs")
    with open(f"{args.save_dir}/args.txt", 'w') as record_txt:
        for key, value in args._get_kwargs():
            record_txt.write(f"{key}={value}\n")


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_epoch(
        preprocessor: torch.nn.Module,
        classifier: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        sig_criterion: torch.nn.Module,
        cls_criterion: torch.nn.Module,
        sig_loss_factor: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: Literal["cuda:0", "cpu"],
        auto_hps: bool,
        cm_length: int = 0,
    ) -> Tuple[float, float, Union[np.ndarray, None]]:
    
    preprocessor.train()

    sig_loss_metric = Metric(50)
    cls_loss_metric = Metric(50)
    loss_metric = Metric(50)
    acc_metric  = Metric(50)
    if cm_length != 0:
        confusion_matrixs: np.ndarray = np.zeros((cm_length, cm_length))
    else:
        confusion_matrixs = None

    if not auto_hps:
        pbar = tqdm(dataloader, desc="[TRAIN]")
    else:
        pbar = dataloader
        print(f"[TRAIN] (length: {pbar.__len__():4d}) ", end='', flush=True)
        start_time = time.time()

    for batch_inputs, batch_truth in pbar:

        preprocessor.zero_grad()
        batch_inputs: torch.Tensor = batch_inputs.to(device)
        batch_truth : torch.Tensor = batch_truth.to(device)

        batch_adjsig: torch.Tensor = preprocessor(batch_inputs)
        batch_pred: torch.Tensor = classifier(batch_adjsig.unsqueeze(1))
        sig_loss: torch.Tensor = sig_criterion(batch_adjsig, batch_inputs) * sig_loss_factor
        cls_loss: torch.Tensor = cls_criterion(batch_pred, batch_truth)
        loss = sig_loss + cls_loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        sig_loss_metric.append(sig_loss.item())
        cls_loss_metric.append(cls_loss.item())
        loss_metric.append(loss.item())

        batch_truth = np.argmax(batch_truth.cpu().detach().numpy(), axis=1)
        batch_pred  = np.argmax(batch_pred.cpu().detach().numpy() , axis=1)
        acc = np.average(batch_truth==batch_pred)
        acc_metric.append(acc)
        
        if cm_length != 0:
            confusion_matrixs += \
                confusion_matrix(batch_truth, batch_pred,
                                 labels=list(range(cm_length)))
        if not auto_hps:
            pbar.set_description(f"[TRAIN] loss: {loss_metric.avg:.5f}, " + \
                                 f"sig_loss: {sig_loss_metric.avg:.5f}, " + \
                                 f"cls_loss: {cls_loss_metric.avg:.5f}, " + \
                                 f"Acc: {acc_metric.avg*100:.3f}%, " + \
                                 f"LR: {get_lr(optimizer):.10f}")
    if auto_hps:
        print(f"loss: {loss_metric.avg:.5f}, " + \
              f"sig_loss: {sig_loss_metric.avg:.5f}, " + \
              f"cls_loss: {cls_loss_metric.avg:.5f}, " + \
              f"Acc: {acc_metric.avg*100:.3f}%, " + \
              f"LR: {get_lr(optimizer):.10f}, " + \
              f"time: {time.time()-start_time:.2f}s", flush=True)
    
    return sig_loss_metric.avg, cls_loss_metric.avg, loss_metric.avg, \
           acc_metric.avg, confusion_matrixs


def valid_epoch(
        preprocessor: torch.nn.Module,
        classifier: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        sig_criterion: torch.nn.Module,
        cls_criterion: torch.nn.Module,
        sig_loss_factor: int,
        device: Literal["cuda:0", "cpu"],
        auto_hps: bool,
        cm_length: int = 0,
    ) -> Tuple[float, float, Union[np.ndarray, None]]:
    
    preprocessor.eval()

    sig_loss_metric = Metric(10000)
    cls_loss_metric = Metric(10000)
    loss_metric = Metric(10000)
    acc_metric  = Metric(10000)
    if cm_length != 0:
        confusion_matrixs = np.zeros((cm_length, cm_length))
    else:
        confusion_matrixs = None

    if not auto_hps:
        pbar = tqdm(dataloader, desc="[VALID]")
    else:
        pbar = dataloader
        print(f"[VALID] (length: {pbar.__len__():4d}) ", end='', flush=True)
        start_time = time.time()

    for batch_inputs, batch_truth in pbar:

        batch_inputs: torch.Tensor = batch_inputs.to(device)
        batch_truth : torch.Tensor = batch_truth.to(device)

        batch_adjsig: torch.Tensor = preprocessor(batch_inputs)
        batch_pred: torch.Tensor = classifier(batch_adjsig.unsqueeze(1))
        sig_loss: torch.Tensor = sig_criterion(batch_adjsig, batch_inputs) * sig_loss_factor
        cls_loss: torch.Tensor = cls_criterion(batch_pred, batch_truth)
        loss = sig_loss + cls_loss
        sig_loss_metric.append(sig_loss.item())
        cls_loss_metric.append(cls_loss.item())
        loss_metric.append(loss.item())

        batch_truth = np.argmax(batch_truth.cpu().detach().numpy(), axis=1)
        batch_pred  = np.argmax(batch_pred.cpu().detach().numpy() , axis=1)
        acc = np.average(batch_truth==batch_pred)
        acc_metric.append(acc)
        
        if cm_length != 0:
            confusion_matrixs += \
                confusion_matrix(batch_truth, batch_pred,
                                 labels=list(range(cm_length)))
        if not auto_hps:
            pbar.set_description(f"[VALID] loss: {loss_metric.avg:.5f}, " + \
                                 f"sig_loss: {sig_loss_metric.avg:.5f}, " + \
                                 f"cls_loss: {cls_loss_metric.avg:.5f}, " + \
                                 f"Acc: {acc_metric.avg*100:.3f}%")
    if auto_hps:
        print(f"loss: {loss_metric.avg:.5f}, " + \
              f"sig_loss: {sig_loss_metric.avg:.5f}, " + \
              f"cls_loss: {cls_loss_metric.avg:.5f}, " + \
              f"Acc: {acc_metric.avg*100:.3f}%, " + \
              f"time: {time.time()-start_time:.2f}s", flush=True)
    
    return sig_loss_metric.avg, cls_loss_metric.avg, loss_metric.avg, \
           acc_metric.avg, confusion_matrixs


def train(args) -> Tuple[float, float, float, float]:
    
    if args.dataset == "BcicIv2a":
        dataset = BcicIv2aDataset(auto_hps=args.auto_hps)
    elif args.dataset == "PhysionetMI":
        dataset = PhysionetMIDataset(auto_hps=args.auto_hps)
    elif args.dataset == "Ofner":
        dataset = Ofner2017Dataset(auto_hps=args.auto_hps)

    train_inputs, train_truths, valid_inputs, valid_truths = \
        dataset.splitted_data_and_label()

    print(train_inputs.shape, train_truths.shape)
    print(valid_inputs.shape, valid_truths.shape)

    my_train_dataset = MyMapDataset(train_inputs, train_truths)
    my_valid_dataset = MyMapDataset(valid_inputs, valid_truths)
    
    my_train_dataLoader = torch.utils.data.DataLoader(
        my_train_dataset, args.batch_size, shuffle=True,
        pin_memory=True, drop_last=False,
        num_workers=args.num_workers)
    my_valid_dataLoader = torch.utils.data.DataLoader(
        my_valid_dataset, args.batch_size, shuffle=True,
        pin_memory=True, drop_last=False,
        num_workers=args.num_workers)

    if args.preprocessor == "LSTM":
        preprocessor = PreLSTM(
            input_size=train_inputs.shape[1],
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout).to(args.device)
    elif args.preprocessor == "Transformer":
        preprocessor = PreTransformer(
            channels=train_inputs.shape[1],
            samples=train_inputs.shape[2],
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout).to(args.device)

    classifier: torch.nn.Module = torch.load(args.classifier_weights)
    classifier = classifier.to(args.device)
    classifier.eval()
    
    sig_criterion: torch.nn.Module = torch.nn.MSELoss()
    cls_criterion: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = \
        torch.optim.Adam(preprocessor.parameters(), lr=args.learning_rate)
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = \
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    backup_files(args)

    best_valid_loss, best_valid_acc = np.inf, 0.0
    train_sig_losses, train_cls_losses, train_losses, train_accs = [], [], [], []
    valid_sig_losses, valid_cls_losses, valid_losses, valid_accs, lrs = [], [], [], [], []
    
    early_stop_counter = 0
    for epoch in range(1, args.epochs+1):
        print(f"{epoch}/{args.epochs}")

        train_results = train_epoch(preprocessor, classifier,
                                    my_train_dataLoader,
                                    sig_criterion, cls_criterion, args.sig_loss_factor,
                                    optimizer, lr_scheduler,
                                    args.device, args.auto_hps,
                                    dataset.class_number)
        valid_results = valid_epoch(preprocessor, classifier,
                                    my_valid_dataLoader,
                                    sig_criterion, cls_criterion, args.sig_loss_factor,
                                    args.device, args.auto_hps,
                                    dataset.class_number)
        
        train_sig_loss, train_cls_loss, train_loss, train_acc, train_cm = train_results
        valid_sig_loss, valid_cls_loss, valid_loss, valid_acc, valid_cm = valid_results

        train_sig_losses.append(train_sig_loss)
        train_cls_losses.append(train_cls_loss)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_sig_losses.append(valid_sig_loss)
        valid_cls_losses.append(valid_cls_loss)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        lrs.append(get_lr(optimizer))

        early_stop_counter += 1
        if valid_loss < best_valid_loss:
            early_stop_counter = 0
            best_valid_loss = valid_loss
            if dataset.class_number != 0:
                plot_confusion_matrix(dataset.class_number, train_cm, 
                                      f"{args.save_dir}/best_valid_loss_train_cm.png",
                                      "Train Confusion Matirx at Best Valid Loss")
                plot_confusion_matrix(dataset.class_number, valid_cm,
                                      f"{args.save_dir}/best_valid_loss_valid_cm.png",
                                      "Valid Confusion Matirx at Best Valid Loss")
            torch.save(preprocessor, f"{args.save_dir}/best_valid_loss.pt")
        if valid_acc > best_valid_acc:
            early_stop_counter = 0
            best_valid_acc = valid_acc
            if dataset.class_number != 0:
                plot_confusion_matrix(dataset.class_number, train_cm,
                                      f"{args.save_dir}/best_valid_acc_train_cm.png",
                                      "Train Confusion Matirx at Best Valid Acc")
                plot_confusion_matrix(dataset.class_number, valid_cm,
                                      f"{args.save_dir}/best_valid_acc_valid_cm.png",
                                      "Valid Confusion Matirx at Best Valid Acc")
            torch.save(preprocessor, f"{args.save_dir}/best_valid_acc.pt")
        
        if (epoch == args.epochs or (epoch > 150 and early_stop_counter >= 50)) and \
           (args.save_plot or args.show_plot):
            history = {"accuracy": train_accs,
                       "sig_loss": train_sig_losses,
                       "cls_loss": train_cls_losses,
                       "loss": train_losses,
                       "val_accuracy": valid_accs,
                       "val_sig_loss": valid_sig_losses,
                       "val_cls_loss": valid_cls_losses,
                       "val_loss": valid_losses,
                       "lr": lrs}
            plot_history_sca(history, f"{args.preprocessor}_{args.classifier}_{args.dataset}",
                             f"{args.save_dir}/history_plot.png",
                             args.save_plot, args.show_plot)
        elif epoch % 50 == 0 and args.save_plot:
            history = {"accuracy": train_accs,
                       "sig_loss": train_sig_losses,
                       "cls_loss": train_cls_losses,
                       "loss": train_losses,
                       "val_accuracy": valid_accs,
                       "val_sig_loss": valid_sig_losses,
                       "val_cls_loss": valid_cls_losses,
                       "val_loss": valid_losses,
                       "lr": lrs}
            plot_history_sca(history, f"{args.preprocessor}_{args.classifier}_{args.dataset}",
                             f"{args.save_dir}/history_plot.png", True, False)
        if epoch > 150 and early_stop_counter >= 50:
            print(f"Early stopping at epoch: {epoch}.", flush=True)
            break

    if not args.auto_hps:
        new_save_dir = args.save_dir.replace("histories_pre_tmp/", '')
        new_save_dir = new_save_dir.split('_', 1)
        new_save_dir = f"histories_pre_tmp/{new_save_dir[0]}_{best_valid_acc*100:.2f}%_{new_save_dir[1]}"
        os.rename(args.save_dir, new_save_dir)
    else:
        new_save_dir = args.save_dir.split('_pt/', 1)
        new_save_dir = f"{new_save_dir[0]}_pt/{best_valid_acc*100:.2f}%_{new_save_dir[1]}"
        os.rename(args.save_dir, new_save_dir)
    return max(train_accs), min(train_losses), max(valid_accs), min(valid_losses)





##### Execution #####
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--preprocessor", type=str, default="Transformer",
        help="The preprocessor to be used. Options: ['LSTM', 'Transformer'].")
    parser.add_argument(
        "-s", "--sig_loss_factor", type=int, default=10,
        help="The x value of `loss = x * sig_loss + cls_loss`.")
    parser.add_argument(
        "-c", "--classifier", type=str, default="EEGNet",
        help="The classifier to be used. " + \
             "Options: ['EEGNet', 'GRU', 'LSTM', 'ATCNet'].")
    parser.add_argument(
        "-cw", "--classifier_weights", type=str,
        default="histories_cls_tmp/04.26-18.46.34_61.55%_pt_EEGNet_BcicIv2a_bs=32_lr=0.0250_ld=0.999910/best_valid_acc.pt",
        help="The path of the weights of the classifier to be used.")
    parser.add_argument(
        "-d", "--dataset", type=str, default="BcicIv2a",
        help="The dataset used for training. " + \
             "Options: ['BcicIv2a', 'PhysionetMI', 'Ofner'].")
    parser.add_argument(
        "-e", "--epochs", type=int, default=600,
        help="The total epochs (iterations) of training.")
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=32,
        help="The batch size of training input.")
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=0.025,
        help="The initial learning rate of the optimizer for training.")
    parser.add_argument(
        "-ld", "--lr_decay", type=float, default=0.99991,
        help="The decay rate of learning rate in each step of training.")
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="The device used to train the model.")
    parser.add_argument(
        "-nw", "--num_workers", type=int, default=1,
        help="The number of CPU workers to use.\n" + \
             "The actual total cost will be doubled due to train and valid dataloaders.\n" + \
             "The actual total cost should be <= the number of your CPU threads.")
    parser.add_argument(
        "--save_plot", type=bool, default=True,
        help="Whether to save the training history plot.")
    parser.add_argument(
        "--show_plot", type=bool, default=False,
        help="Whether to show the training history plot.")
    parser.add_argument(
        "--auto_hps", type=bool, default=False,
        help="Whether doing the auto hyperparameter searching.")

    args = parser.parse_args()

    if args.preprocessor == "LSTM":
        args.num_layers  = 3
        args.hidden_size = 64
        args.dropout     = 0.5
    elif args.preprocessor == "Transformer":
        args.num_layers = 1
        args.num_heads  = 3
        args.ffn_dim    = 64
        args.dropout    = 0.5

    args.save_dir = time.strftime("histories_pre_tmp/%m.%d-%H.%M.%S_pt")
    args.save_dir += f"_{args.preprocessor}_{args.classifier}_{args.dataset}"
    args.save_dir += f"_slf={args.sig_loss_factor:03d}"
    args.save_dir += f"_bs={args.batch_size:03d}"
    args.save_dir += f"_lr={args.learning_rate:.4f}"
    args.save_dir += f"_ld={args.lr_decay:.6f}"
    if args.preprocessor == "LSTM":
        args.save_dir += f"_nl={args.num_layers}_hs={args.hidden_size:03d}_do={args.dropout:.2f}"
    elif args.preprocessor == "Transformer":
        args.save_dir += f"_nl={args.num_layers}_nh={args.num_heads:02d}"
        args.save_dir += f"_fd={args.ffn_dim:03d}_do={args.dropout:.2f}"

    train(args)