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
from typing import Union, Tuple, List, Literal
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, Subset
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
warnings.filterwarnings(action="ignore", category=FutureWarning)

import utils
# from models_pytorch.EEGNet import EEGNet
from torcheeg.models import EEGNet, GRU, LSTM, ATCNet
from utils import Metric, plot_confusion_matrix, plot_history
from libs.dataset import BcicIv2aDataset, PhysionetMIDataset, Ofner2017Dataset  # , InnerSpeechDataset





##### Classes #####
class MyMapDataset(Dataset):
    def __init__(self,
                 inputs          : Union[list, np.ndarray],
                 truths          : Union[list, np.ndarray],
                 # original_signal : bool = True,
                 mixed_signal    : bool = False,
                 mixing_source   : int  = 2,
                 mixing_duplicate: int  = 1) -> None:
        
        # self.original_signal = original_signal
        # self.mixed_signal = mixed_signal
        # self.mixing_source = mixing_source
        # self.mixing_duplicate = mixing_duplicate
        # if self.original_signal and self.mixed_signal:
        #     self.original_chance = random.random() < (1 / self.mixing_duplicate+1)

        if mixed_signal:
            truths = np.argmax(truths, 1)
            group = { tth: [] for tth in set(truths) }
            for ipt, tth in zip(inputs, truths):
                group[tth].append(ipt)
            group[tth] = np.array(group[tth])
            for tth in group:
                ori_sig = group[tth].copy()
                shf_sig = group[tth].copy()
                for _ in range(mixing_duplicate):
                    mix_factors = np.random.random((mixing_source, len(ori_sig)))
                    mix_factors /= np.sum(mix_factors, axis=0)
                    mixed_signal = np.reshape(mix_factors[0], (len(ori_sig),1,1,1)) * ori_sig
                    for mf in mix_factors[1:]:
                        random.shuffle(shf_sig)
                        mixed_signal += np.reshape(mf, (len(ori_sig),1,1,1))*shf_sig
                    group[tth] = np.concatenate([group[tth], mixed_signal])
            inputs = np.concatenate(list(group.values()))
            truths = np.concatenate([ np.array([[k==i for i in range(4)]]*len(v)) for k, v in group.items() ])

        self.inputs = torch.Tensor(inputs)
        self.truths = torch.Tensor(truths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.truths[index]
        # if not self.mixed_signal:
        #     return self.inputs[index], self.truths[index]
        # elif self.original_signal and self.mixed_signal and self.original_chance:
        #     return self.inputs[index], self.truths[index]
        # else:
        #     truth = self.truths[index]
        #     return self.inputs[index], 

    def __len__(self) -> int:
        return len(self.inputs)





##### Functions #####
def backup_files(args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir)
    shutil.copy(__file__, args.save_dir)
    shutil.copy(utils.__file__, args.save_dir)
    shutil.copytree("libs", f"{args.save_dir}/libs")
    # shutil.copy(models_pytorch.EEGNet.__file__, args.save_dir)
    with open(f"{args.save_dir}/args.txt", 'w') as record_txt:
        for key, value in args._get_kwargs():
            record_txt.write(f"{key}={value}\n")


def split_data(
        inputs: Union[list, np.ndarray],
        truths: Union[list, np.ndarray]
    ) -> Tuple[ Union[list, np.ndarray], Union[list, np.ndarray],
                Union[list, np.ndarray], Union[list, np.ndarray] ]:
    dataset_length = len(inputs)
    train_dataset_length = int(dataset_length*0.8)
    data = list(zip(inputs, truths))
    random.shuffle(data)
    train_data, valid_data = data[:train_dataset_length], data[train_dataset_length:]
    train_inputs, train_truths = np.array([ td[0] for td in train_data ]), np.array([ td[1] for td in train_data ])
    valid_inputs, valid_truths = np.array([ vd[0] for vd in valid_data ]), np.array([ vd[1] for vd in valid_data ])
    return train_inputs, train_truths, valid_inputs, valid_truths


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
    

def train_epoch(
        model: torch.nn.Module,
        dataloader: List[Tuple[torch.Tensor, torch.Tensor]],
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: Literal["cuda:0", "cpu"],
        auto_hps: bool,
        cm_length: int = 0,
        weight: list = None,
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

    if not auto_hps:
        pbar = tqdm(dataloader, desc="[TRAIN]")  # , ascii=True)
    else:
        pbar = dataloader
        print("[TRAIN] ", end='')
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
        if not auto_hps:
            pbar.set_description(f"[TRAIN] loss: {loss_metric.avg:.5f}, " + \
                                 f"Acc: {acc_metric.avg*100:.3f}%, " + \
                                 f"LR: {get_lr(optimizer):.10f}")
    if auto_hps:
        print(f"loss: {loss_metric.avg:.5f}, " + \
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
        auto_hps: bool,
        cm_length: int = 0,
        weight: list = None,
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

    if not auto_hps:
        pbar = tqdm(dataloader, desc="[VALID]")  # , ascii=True)
    else:
        pbar = dataloader
        print("[VALID] ", end='')
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
        if not auto_hps:
            pbar.set_description(f"[VALID] loss: {loss_metric.avg:.5f}, " + \
                                 f"Acc: {acc_metric.avg*100:.3f}%, ")
    
    if auto_hps:
        print(f"loss: {loss_metric.avg:.5f}, " + \
              f"Acc: {acc_metric.avg*100:.3f}%, ")

    # with np.printoptions(linewidth=150, formatter={'float': '{:5.03f}'.format, 'int': '{:2} '.format}):
    #     print("pred :", batch_pred)
    #     print("truth:", batch_truth)
    #     print("corct:", np.array(['X', 'O'])[np.uint8(batch_pred==batch_truth)])
    
    return loss_metric.avg, acc_metric.avg, confusion_matrixs


def train(args) -> Tuple[List[float], List[float], List[float], List[float]]:
    # assert args.dataset in ["BcicIv2a", "PhysionetMI", "Ofner", "InnerSpeech"], \
    assert args.dataset in ["BcicIv2a", "PhysionetMI", "Ofner"], \
        "Invalid value for parameter 'dataset'."

    if args.dataset == "BcicIv2a":
        dataset = BcicIv2aDataset(auto_hps=args.auto_hps)
    elif args.dataset == "PhysionetMI":
        dataset = PhysionetMIDataset(auto_hps=args.auto_hps)
    elif args.dataset == "Ofner":
        dataset = Ofner2017Dataset(auto_hps=args.auto_hps)
    # elif args.dataset == "InnerSpeech":
    #     dataset = InnerSpeechDataset()

    train_inputs, train_truths, valid_inputs, valid_truths = \
        dataset.splitted_data_and_label()
    train_inputs = np.expand_dims(train_inputs, axis=1)
    valid_inputs = np.expand_dims(valid_inputs, axis=1)

    print(train_inputs.shape, train_truths.shape)
    print(valid_inputs.shape, valid_truths.shape)

    my_train_dataset = MyMapDataset(train_inputs, train_truths,
                                    mixed_signal=args.mixed_signal,
                                    mixing_source=args.mixing_source,
                                    mixing_duplicate=args.mixing_duplicate)
    my_valid_dataset = MyMapDataset(valid_inputs, valid_truths)
    
    my_train_dataLoader = torch.utils.data.DataLoader(
        my_train_dataset, args.batch_size, shuffle=True,
        pin_memory=True, drop_last=False,
        num_workers=args.num_workers, persistent_workers=True)
    my_valid_dataLoader = torch.utils.data.DataLoader(
        my_valid_dataset, args.batch_size, shuffle=True,
        pin_memory=True, drop_last=False,
        num_workers=args.num_workers, persistent_workers=True)

    # print("ground_truth_count:", my_train_dataset.ground_truth_count)
    # train_btc_avg      = sum(my_train_dataset.ground_truth_count) / len(my_train_dataset.ground_truth_count)
    # train_weight       = [ (train_btc_avg/btc) for btc in my_train_dataset.ground_truth_count ]
    # train_weight_torch = torch.from_numpy(np.array(train_weight)).float().to(args.device)
    # valid_btc_avg      = sum(my_valid_dataset.ground_truth_count) / len(my_valid_dataset.ground_truth_count)
    # valid_weight       = [ (valid_btc_avg/btc) for btc in my_valid_dataset.ground_truth_count ]

    if args.model == "EEGNet":
        # model = EEGNet(
        #             nb_classes=dataset.class_number,
        #             Chans=train_inputs.shape[2], Samples=train_inputs.shape[3],
        #             dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
        #             dropoutType="Dropout").to(args.device)
        model = EEGNet(
            kernel_1=32,
            kernel_2=16,
            dropout=0.5,
            F1=8,
            F2=16,
            D=2,
            chunk_size=train_inputs.shape[3],
            num_electrodes=train_inputs.shape[2],
            num_classes=dataset.class_number).to(args.device)
    elif args.model == "GRU":
        model = GRU(
            hid_channels=args.hid_channels,
            num_electrodes=train_inputs.shape[2],
            num_classes=dataset.class_number).to(args.device)
    elif args.model == "LSTM":
        model = LSTM(
            hid_channels=args.hid_channels,
            num_electrodes=train_inputs.shape[2],
            num_classes=dataset.class_number).to(args.device)
    elif args.model == "ATCNet":
        model = ATCNet(
            num_windows=args.num_windows,
            conv_pool_size=7,
            F1=16,
            D=2,
            tcn_kernel_size=4,
            tcn_depth=2,
            num_classes=dataset.class_number,
            num_electrodes=train_inputs.shape[2],
            chunk_size=train_inputs.shape[3]).to(args.device)
    
    criterion: torch.nn.Module = \
        torch.nn.CrossEntropyLoss()  # weight=train_weight_torch)
    optimizer: torch.optim.Optimizer = \
        torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = \
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    backup_files(args)

    tensorboard = SummaryWriter(args.save_dir)
    best_valid_loss, best_valid_acc = np.inf, 0.0
    train_losses, train_accs, valid_losses, valid_accs, lrs = [], [], [], [], []
    early_stop_counter = 0
    for epoch in range(1, args.epochs+1):
        print(f"{epoch}/{args.epochs}")

        train_results = train_epoch(model, my_train_dataLoader,
                                    criterion, optimizer, lr_scheduler,
                                    args.device, args.auto_hps,
                                    dataset.class_number)  # , train_weight)
        valid_results = valid_epoch(model, my_valid_dataLoader,
                                    criterion, args.device, args.auto_hps,
                                    dataset.class_number)  # , valid_weight)
        
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
            print(f"Early stopping at epoch: {epoch}.")
            break

    tensorboard.close()
    if not args.auto_hps:
        new_save_dir = args.save_dir.replace("histories_tmp/", '')
        new_save_dir = new_save_dir.split('_', 1)
        new_save_dir = f"histories_tmp/{new_save_dir[0]}_{best_valid_acc*100:.2f}%_{new_save_dir[1]}"
        os.rename(args.save_dir, new_save_dir)
    else:
        new_save_dir = args.save_dir.split('_pt/', 1)
        new_save_dir = f"{new_save_dir[0]}_pt/{best_valid_acc*100:.2f}%_{new_save_dir[1]}"
        os.rename(args.save_dir, new_save_dir)
    return train_accs, train_losses, valid_accs, valid_losses





##### Execution #####
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="ATCNet",
        help="The model to be trained. " + \
             "Options: ['EEGNet', 'GRU', 'LSTM', 'ATCNet'].")
    parser.add_argument(
        "-d", "--dataset", type=str, default="BcicIv2a",
        help="The dataset used for training. " + \
             "Options: ['BcicIv2a', 'PhysionetMI', 'Ofner'].")
             # "Options: ['BcicIv2a', 'PhysionetMI', 'Ofner', 'InnerSpeech'].")
    # parser.add_argument(
    #     "-os", "--original-signal", type=bool, default=True,
    #     help="Whether to use the original signal for training.")
    parser.add_argument(
        "-mx", "--mixed-signal", type=bool, default=False,
        help="Whether to use the mixed signal for training.")
    parser.add_argument(
        "-ms", "--mixing-source", type=int, default=2,
        help="The amount of signal source used to mix one mixed signal.")
    parser.add_argument(
        "-md", "--mixing-duplicate", type=int, default=2,
        help="The amount of mixed signal for training.")
    parser.add_argument(
        "-e", "--epochs", type=int, default=1000,
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
    # parser.add_argument(
    #     "-sp", "--save_dir", type=str, default=None,
    #     help="The path to save all history files.")
    parser.add_argument(
        "-s", "--save_plot", type=bool, default=True,
        help="Whether to save the training history plot.")
    parser.add_argument(
        "--show_plot", type=bool, default=False,
        help="Whether to show the training history plot.")

    args = parser.parse_args()
    # assert args.original_signal and args.mixed_signal, \
    #     "Either 'original_signal' or 'mixed_signal' must be True."
    if args.mixed_signal:
        assert args.mixing_duplicate >= 1, \
            "Parameter 'mixing_duplicate' must be >= 1 when using 'mixed_signal'."
        assert args.mixing_source >= 2, \
            "Parameter 'mixing_source' must be >= 2 when using 'mixed_signal'."

    args.save_dir = time.strftime("histories_tmp/%m.%d-%H.%M.%S_pt")
    args.save_dir += f"_{args.model}_{args.dataset}"
    args.save_dir += f"_bs={args.batch_size}"
    args.save_dir += f"_lr={args.learning_rate:.4f}"
    args.save_dir += f"_ld={args.lr_decay:.6f}"
    args.save_dir += f"_os"
    # if args.original_signal: args.save_dir += f"_os"
    if args.mixed_signal: args.save_dir += f"_m{args.mixing_source}x{args.mixing_duplicate}"

    args.auto_hps = False

    if args.model == "EEGNet":
        train(args)