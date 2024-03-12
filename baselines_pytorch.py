##### Libraries #####
import os
import time
import torch
import utils
import shutil
import argparse
import numpy as np
import pandas as pd
import seaborn as sn
import torch.utils.data
import models_pytorch.EEGNet
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Union, Tuple, List, Literal
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, Subset
from torch.utils.tensorboard import SummaryWriter





##### Classes #####
class MyMapDataset(Dataset):
    def __init__(self, inputs: Union[list, np.ndarray], truths: Union[list, np.ndarray]) -> None:
        self.inputs = torch.Tensor(inputs)
        self.truths = torch.Tensor(truths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.truths[index]

    def __len__(self) -> int:
        return len(self.inputs)


class Metric():
    def __init__(self, length: int) -> None:
        self.length = length
        self.values = []

    def append(self, value) -> None:
        self.values.append(value)
        if len(self.values) > self.length: self.values.pop(0)
        return

    @ property
    def avg(self) -> float:
        return np.average(self.values)





##### Functions #####
def backup_files(args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir)
    shutil.copy(__file__, args.save_dir)
    shutil.copy(utils.__file__, args.save_dir)
    shutil.copy(models_pytorch.EEGNet.__file__, args.save_dir)
    with open(f"{args.save_dir}/args.txt", 'w') as record_txt:
        for key, value in args._get_kwargs():
            record_txt.write(f"{key}={value}\n")


def split_datasets(dataset: Dataset) -> Tuple[Subset, Subset]:
    dataset_length = len(dataset)
    train_dataset_length = int(dataset_length*0.8)
    valid_dataset_length = dataset_length - train_dataset_length
    train_dataset, valid_dataset = \
        torch.utils.data.random_split(
            dataset, [train_dataset_length, valid_dataset_length],
            generator=torch.Generator().manual_seed(0))
    return train_dataset, valid_dataset


def get_weighted_acc(pred, truth, weight) -> float:
    weighted_acc = 0.0
    # assert np.max(np.max(pred), np.max(truth)) <= len(weight)
    for wid, wt in enumerate(weight):
        weighted_acc += (np.logical_and(pred==wid, pred==truth).sum()/np.max((1, (truth==wid).sum()))) * wt
    return weighted_acc


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    

def plot_confusion_matrix(
        cm_length: int,
        cm: np.ndarray,
        filename: str,
        title: str,
    ) -> None:
    cm_df = pd.DataFrame(cm, index=list(range(cm_length)), columns=list(range(cm_length)))
    plt.figure(figsize=(6, 5))
    cm_image:plt.Axes = sn.heatmap(cm_df, annot=True)
    cm_image.set_xlabel("prediction", fontsize=10)
    cm_image.set_ylabel("truth", fontsize=10)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return


def train_epoch(
        model: torch.nn.Module,
        dataloader: List[Tuple[torch.Tensor, torch.Tensor]],
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: Literal["cuda:0", "cpu"],
        weight: list = None,
        cm_length: int = 0
    ) -> Tuple[float, float, Union[np.ndarray, None]]:
    
    model = model.train()

    loss_metric = Metric(50)
    acc_metric  = Metric(50)
    if weight is not None:
        weight = (np.array(weight)/sum(weight)).tolist()
    if cm_length != 0:
        confusion_matrixs: np.ndarray = np.zeros((cm_length, cm_length))
    else:
        confusion_matrixs = None

    pbar = tqdm(dataloader, desc="[TRAIN]")  # , ascii=True)
    for batch_inputs, batch_truth in pbar:

        model.zero_grad()
        batch_inputs = batch_inputs.to(device)
        batch_truth  = batch_truth.to(device)

        batch_pred: torch.Tensor = model(batch_inputs)
        loss: torch.Tensor = criterion(batch_pred, batch_truth)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        loss_metric.append(loss.item())

        batch_truth = np.argmax(batch_truth.cpu().detach().numpy(), axis=1)
        batch_pred  = np.argmax(batch_pred.cpu().detach().numpy() , axis=1)
        if weight is None:
            acc = np.average(batch_truth==batch_pred)
        else:
            acc = get_weighted_acc(batch_pred, batch_truth, weight)
        acc_metric.append(acc)
        
        if cm_length != 0:
            confusion_matrixs += \
                confusion_matrix(batch_truth, batch_pred,
                                 labels=list(range(cm_length)))  # , sample_weight=weight)
        pbar.set_description(f"[TRAIN] loss: {loss_metric.avg:.5f}, " + \
                             f"Acc: {acc_metric.avg*100:.3f}%, " + \
                             f"LR: {get_lr(optimizer):.10f}")
        
    # with np.printoptions(linewidth=150, formatter={'float': '{:5.03f}'.format, 'int': '{:2} '.format}):
    #     print("pred :", batch_pred)
    #     print("truth:", batch_truth)
    #     print("corct:", np.array(['X', 'O'])[np.uint8(batch_pred==batch_truth)])
    
    return loss_metric.avg, acc_metric.avg, confusion_matrixs


def valid_epoch(
        model: torch.nn.Module,
        dataloader: List[Tuple[torch.Tensor, torch.Tensor]],
        criterion: torch.nn.Module,
        device: Literal["cuda:0", "cpu"],
        weight: list = None,
        cm_length: int = 0
    ) -> Tuple[float, float, Union[np.ndarray, None]]:
    
    model = model.eval()

    loss_metric = Metric(10000)
    acc_metric  = Metric(10000)
    if weight is not None:
        weight = (np.array(weight)/sum(weight)).tolist()
    if cm_length != 0:
        confusion_matrixs = np.zeros((cm_length, cm_length))
    else:
        confusion_matrixs = None

    pbar = tqdm(dataloader, desc="[VALID]")  # , ascii=True)
    for batch_inputs, batch_truth in pbar:

        model.zero_grad()
        batch_inputs = batch_inputs.to(device)
        batch_truth  = batch_truth.to(device)

        batch_pred: torch.Tensor = model(batch_inputs)
        loss: torch.Tensor = criterion(batch_pred, batch_truth)
        loss_metric.append(loss.item())

        batch_truth = np.argmax(batch_truth.cpu().detach().numpy(), axis=1)
        batch_pred  = np.argmax(batch_pred.cpu().detach().numpy() , axis=1)
        if weight is None:
            acc = np.average(batch_truth==batch_pred)
        else:
            acc = get_weighted_acc(batch_pred, batch_truth, weight)
        acc_metric.append(acc)
        
        if cm_length != 0:
            confusion_matrixs += \
                confusion_matrix(batch_truth, batch_pred,
                                 labels=list(range(cm_length)))  # , sample_weight=weight)
        pbar.set_description(f"[VALID] loss: {loss_metric.avg:.5f}, " + \
                             f"Acc: {acc_metric.avg*100:.3f}%, ")
        
    # with np.printoptions(linewidth=150, formatter={'float': '{:5.03f}'.format, 'int': '{:2} '.format}):
    #     print("pred :", batch_pred)
    #     print("truth:", batch_truth)
    #     print("corct:", np.array(['X', 'O'])[np.uint8(batch_pred==batch_truth)])
    
    return loss_metric.avg, acc_metric.avg, confusion_matrixs


def baseline_EEGNet(args):
    assert args.dataset in ["BCIC-IV-2a"], "Invalid value for parameter 'dataset'."

    tensorboard = SummaryWriter(args.save_dir)
    if args.dataset == "BCIC-IV-2a":
        dataset = utils.BcicIv2aDataset()  # l_freq=4
        inputs = np.concatenate([ v for v in dataset.data.values()   ], axis=0)
        inputs = np.expand_dims(inputs, axis=1)
        truths = np.concatenate([ v for v in dataset.labels.values() ], axis=0)
        cm_length = len(set(truths))
        truths = np.array([ [ v==l for l in range(cm_length) ] for v in truths ])

    my_dataset = MyMapDataset(inputs, truths)
    my_train_dataset, my_valid_dataset = split_datasets(my_dataset)
    my_train_dataLoader = torch.utils.data.DataLoader(
        my_train_dataset, args.batch_size, shuffle=True,
        pin_memory=True, drop_last=False,
        num_workers=args.num_workers, persistent_workers=True,
    )
    my_valid_dataLoader = torch.utils.data.DataLoader(
        my_valid_dataset, args.batch_size, shuffle=True,
        pin_memory=True, drop_last=False,
        num_workers=args.num_workers, persistent_workers=True,
    )

    # print("ground_truth_count:", my_train_dataset.ground_truth_count)
    # train_btc_avg      = sum(my_train_dataset.ground_truth_count) / len(my_train_dataset.ground_truth_count)
    # train_weight       = [ (train_btc_avg/btc) for btc in my_train_dataset.ground_truth_count ]
    # train_weight_torch = torch.from_numpy(np.array(train_weight)).float().to(args.device)
    # valid_btc_avg      = sum(my_valid_dataset.ground_truth_count) / len(my_valid_dataset.ground_truth_count)
    # valid_weight       = [ (valid_btc_avg/btc) for btc in my_valid_dataset.ground_truth_count ]

    model_8_2 = models_pytorch.EEGNet.EEGNet(
                    nb_classes=4, Chans=inputs.shape[2], Samples=inputs.shape[3],
                    dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
                    dropoutType="Dropout").to(args.device)
    criterion: torch.nn.Module = \
        torch.nn.CrossEntropyLoss()  # weight=train_weight_torch)
    optimizer: torch.optim.Optimizer = \
        torch.optim.Adam(model_8_2.parameters(), lr=args.learning_rate)
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = \
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    best_valid_loss, best_valid_acc = np.inf, 0.0
    train_losses, train_accs, valid_losses, valid_accs, lrs = [], [], [], [], []
    for epoch in range(1, args.epochs+1):
        print(f"{epoch}/{args.epochs}")

        train_results = train_epoch(model_8_2, my_train_dataLoader, criterion, optimizer,
                                    lr_scheduler, args.device, cm_length=cm_length)  # , train_weight)
        valid_results = valid_epoch(model_8_2, my_valid_dataLoader, criterion, args.device,
                                    cm_length=cm_length)  # , valid_weight)
        
        train_loss, train_acc, train_cm = train_results
        valid_loss, valid_acc, valid_cm = valid_results

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        lrs.append(get_lr(optimizer))
        
        tensorboard.add_scalar("0_Losses+LR/0_Train",  train_loss,        epoch)
        tensorboard.add_scalar("0_Losses+LR/1_Valid",  valid_loss,        epoch)
        tensorboard.add_scalar("0_Losses+LR/2_LR",     get_lr(optimizer), epoch)
        tensorboard.add_scalar("1_Accuracies/0_Train", train_acc,         epoch)
        tensorboard.add_scalar("1_Accuracies/1_Valid", valid_acc,         epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if cm_length != 0:
                plot_confusion_matrix(cm_length, train_cm, 
                                      f"{args.save_dir}/best_valid_loss_train_cm.png",
                                      "Train Confusion Matirx at Best Valid Loss")
                plot_confusion_matrix(cm_length, valid_cm,
                                      f"{args.save_dir}/best_valid_loss_valid_cm.png",
                                      "Valid Confusion Matirx at Best Valid Loss")
            torch.save(model_8_2, f"{args.save_dir}/best_valid_loss.pt")
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            if cm_length != 0:
                plot_confusion_matrix(cm_length, train_cm,
                                      f"{args.save_dir}/best_valid_acc_train_cm.png",
                                      "Train Confusion Matirx at Best Valid Acc")
                plot_confusion_matrix(cm_length, valid_cm,
                                      f"{args.save_dir}/best_valid_acc_valid_cm.png",
                                      "Valid Confusion Matirx at Best Valid Acc")
            torch.save(model_8_2, f"{args.save_dir}/best_valid_acc.pt")

    history = {
        "accuracy": train_accs,
        "val_accuracy": valid_accs,
        "loss": train_losses,
        "val_loss": valid_losses,
        "lr": lrs,
    }
    utils.plot_history(history, "EEGNet",
                       f"{args.save_dir}/history_plot.png", args.save_plot, args.show_plot)
    tensorboard.close()
    os.rename(args.save_dir, f"{args.save_dir}_{best_valid_acc*100:.2f}%")
    return





##### Execution #####
if __name__ == "__main__":

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
        "-e", "--epochs", type=int, default=300,
        help="The total epochs (iterations) of training."
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=32,
        help="The batch size of training input."
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=8e-3,
        help="The initial learning rate of the optimizer for training."
    )
    parser.add_argument(
        "-ld", "--lr_decay", type=float, default=0.9999,
        help="The decay rate of learning rate in each step of training."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="The device used to train the model."
    )
    parser.add_argument(
        "-nw", "--num_workers", type=int, default=8,
        help="The number of CPU workers to use.\n" + \
             "The total cost will be double due to train and valid dataloaders.\n" + \
             "The total cost should be <= the number of your CPU threads."
    )
    parser.add_argument(
        "-sp", "--save_dir", type=str, default=None,
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
        args.save_dir = time.strftime("histories/%m.%d-%H.%M.%S")
        args.save_dir += f"_{args.model}_{args.dataset}"
        args.save_dir += f"_bs={args.batch_size}"
        args.save_dir += f"_lr={args.learning_rate}"
        args.save_dir += f"_ld={args.lr_decay}"

    backup_files(args)

    if args.model == "EEGNet":
        baseline_EEGNet(args)