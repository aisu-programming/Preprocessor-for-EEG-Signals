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
from torcheeg.models import EEGNet, ATCNet
from torch.utils.data import Dataset

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
warnings.filterwarnings(action="ignore", category=FutureWarning)

import utils
from utils import Metric, plot_confusion_matrix, plot_history
from libs.dataset import BcicIv2aDataset, PhysionetMIDataset, Ofner2017Dataset
from models_pytorch.gru import GRU
from models_pytorch.lstm import LSTM





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
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: Literal["cuda:0", "cpu", "mps:0"],
        auto_hps: bool,
        cm_length: int = 0,
    ) -> Tuple[float, float, Union[np.ndarray, None]]:
    
    model.train()

    loss_metric = Metric(50)
    acc_metric  = Metric(50)
    if cm_length != 0:
        confusion_matrixs: np.ndarray = np.zeros((cm_length, cm_length))
    else:
        confusion_matrixs = None

    if not auto_hps:
        pbar = tqdm(dataloader, desc="[TRAIN]")  # , ascii=True)
    else:
        pbar = dataloader
        print(f"[TRAIN] (length: {pbar.__len__():4d}) ", end='', flush=True)
        start_time = time.time()
        
    for batch_inputs, batch_truth in pbar:

        batch_inputs: torch.Tensor = batch_inputs.to(device)
        batch_truth : torch.Tensor = batch_truth.to(device)

        model.zero_grad()
        batch_pred: torch.Tensor = model(batch_inputs)
        loss: torch.Tensor = criterion(batch_pred, batch_truth)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
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
                                 f"Acc: {acc_metric.avg*100:.3f}%, " + \
                                 f"LR: {get_lr(optimizer):.10f}")
    if auto_hps:
        print(f"loss: {loss_metric.avg:.5f}, " + \
              f"Acc: {acc_metric.avg*100:.3f}%, " + \
              f"LR: {get_lr(optimizer):.10f}, " + \
              f"time: {time.time()-start_time:.2f}s", flush=True)
    
    return loss_metric.avg, acc_metric.avg, confusion_matrixs


def valid_epoch(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: Literal["cuda:0", "cpu", "mps:0"],
        auto_hps: bool,
        cm_length: int = 0,
    ) -> Tuple[float, float, Union[np.ndarray, None]]:
    
    model.eval()

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

        batch_pred: torch.Tensor = model(batch_inputs)
        loss: torch.Tensor = criterion(batch_pred, batch_truth)
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
                                 f"Acc: {acc_metric.avg*100:.3f}%")
    
    if auto_hps:
        print(f"loss: {loss_metric.avg:.5f}, " + \
              f"Acc: {acc_metric.avg*100:.3f}%, " + \
              f"time: {time.time()-start_time:.2f}s", flush=True)
    
    return loss_metric.avg, acc_metric.avg, confusion_matrixs


def train(args) -> Tuple[float, float, float, float]:
    assert args.dataset in ["BcicIv2a", "PhysionetMI", "Ofner"], \
        "Invalid value for parameter 'dataset'."
    
    backup_files(args)

    if args.dataset == "BcicIv2a":
        dataset = BcicIv2aDataset(auto_hps=args.auto_hps)
    elif args.dataset == "PhysionetMI":
        dataset = PhysionetMIDataset(auto_hps=args.auto_hps)
    elif args.dataset == "Ofner":
        dataset = Ofner2017Dataset(auto_hps=args.auto_hps)

    train_inputs, train_truths, valid_inputs, valid_truths = \
        dataset.splitted_data_and_label()
    if args.model in ["EEGNet", "ATCNet"]:
        train_inputs = np.expand_dims(train_inputs, axis=1)
        valid_inputs = np.expand_dims(valid_inputs, axis=1)

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

    if args.model == "EEGNet":
        model = EEGNet(
            kernel_1=args.kernel_1,
            kernel_2=args.kernel_2,
            dropout=args.dropout,
            F1=args.F1,
            F2=args.F2,
            D=args.D,
            chunk_size=train_inputs.shape[3],
            num_electrodes=train_inputs.shape[2],
            num_classes=dataset.class_number).to(args.device)
    elif args.model == "GRU":
        model = GRU(
            hid_channels=args.hid_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_electrodes=train_inputs.shape[1],
            num_classes=dataset.class_number).to(args.device)
    elif args.model == "LSTM":
        model = LSTM(
            hid_channels=args.hid_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_electrodes=train_inputs.shape[1],
            num_classes=dataset.class_number).to(args.device)
    elif args.model == "ATCNet":
        model = ATCNet(
            num_windows=args.num_windows,
            conv_pool_size=args.conv_pool_size,
            F1=args.F1,
            D=args.D,
            tcn_kernel_size=args.tcn_kernel_size,
            tcn_depth=args.tcn_depth,
            num_classes=dataset.class_number,
            num_electrodes=train_inputs.shape[2],
            chunk_size=train_inputs.shape[3]).to(args.device)
    
    criterion: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = \
        torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = \
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    best_valid_loss, best_valid_acc = np.inf, 0.0
    train_losses, train_accs, valid_losses, valid_accs, lrs = [], [], [], [], []
    early_stop_counter = 0
    for epoch in range(1, args.epochs+1):
        print(f"{epoch}/{args.epochs}")

        train_results = train_epoch(model, my_train_dataLoader,
                                    criterion, optimizer, lr_scheduler,
                                    args.device, args.auto_hps,
                                    dataset.class_number)
        valid_results = valid_epoch(model, my_valid_dataLoader,
                                    criterion, args.device, args.auto_hps,
                                    dataset.class_number)
        
        train_loss, train_acc, train_cm = train_results
        valid_loss, valid_acc, valid_cm = valid_results

        train_losses.append(train_loss)
        train_accs.append(train_acc)
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
            torch.save(model, f"{args.save_dir}/best_valid_loss.pt")
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
            torch.save(model, f"{args.save_dir}/best_valid_acc.pt")
        
        if (epoch == args.epochs or (epoch > 500 and early_stop_counter >= 100)) and \
           (args.save_plot or args.show_plot):
            history = {"accuracy": train_accs,
                       "val_accuracy": valid_accs,
                       "loss": train_losses,
                       "val_loss": valid_losses,
                       "lr": lrs}
            plot_history(history, args.model,
                         f"{args.save_dir}/history_plot.png",
                         args.save_plot, args.show_plot)
        elif epoch % 50 == 0 and args.save_plot:
            history = {"accuracy": train_accs,
                       "val_accuracy": valid_accs,
                       "loss": train_losses,
                       "val_loss": valid_losses,
                       "lr": lrs}
            plot_history(history, args.model,
                         f"{args.save_dir}/history_plot.png", True, False)
        if epoch > 500 and early_stop_counter >= 100:
            print(f"Early stopping at epoch: {epoch}.", flush=True)
            break

    if not args.auto_hps:
        new_save_dir = args.save_dir.replace("histories_cls_tmp/", '')
        new_save_dir = new_save_dir.split('_', 1)
        new_save_dir = f"histories_cls_tmp/{new_save_dir[0]}_{best_valid_acc*100:.2f}%_{new_save_dir[1]}"
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
        "-m", "--model", type=str, default="GRU",
        help="The model to be trained. " + \
             "Options: ['EEGNet', 'GRU', 'LSTM', 'ATCNet'].")
    parser.add_argument(
        "-d", "--dataset", type=str, default="BcicIv2a",
        help="The dataset used for training. " + \
             "Options: ['BcicIv2a', 'PhysionetMI', 'Ofner'].")
    parser.add_argument(
        "-e", "--epochs", type=int, default=1000,
        help="The total epochs (iterations) of training.")
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=32,
        help="The batch size of training input.")
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=0.08,
        help="The initial learning rate of the optimizer for training.")
    parser.add_argument(
        "-ld", "--lr_decay", type=float, default=0.99987,
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

    if args.model == "EEGNet":
        args.kernel_1 = 32
        args.kernel_2 = 16
        args.dropout = 0.5
        args.F1 = 8
        args.F2 = 16
        args.D = 2
    elif args.model in ["GRU", "LSTM"]:
        args.num_layers = 2
        args.hid_channels = 64
        args.dropout = 0.5
    elif args.model == "ATCNet":
        args.num_windows = 3
        args.conv_pool_size = 7
        args.F1 = 16
        args.D = 2
        args.tcn_kernel_size = 4
        args.tcn_depth = 2

    args.save_dir = time.strftime("histories_cls_tmp/%m.%d-%H.%M.%S_pt")
    args.save_dir += f"_{args.model}_{args.dataset}"
    args.save_dir += f"_bs={args.batch_size:03d}"
    args.save_dir += f"_lr={args.learning_rate:.4f}"
    args.save_dir += f"_ld={args.lr_decay:.6f}"
    if args.model == "EEGNet":
        args.save_dir += f"_k1={args.kernel_1}_k2={args.kernel_2}"
        args.save_dir += f"_do={args.dropout:.02f}"
    elif args.model in ["GRU", "LSTM"]:
        args.save_dir += f"_nl={args.num_layers}_hc={args.hid_channels:03d}"
        args.save_dir += f"_do={args.dropout:.02f}"
    elif args.model == "ATCNet":
        args.save_dir += f"_nw={args.num_windows}_cps={args.conv_pool_size}"

    train(args)