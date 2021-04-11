from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from src.data_manager import DataManager
from src.net import Net
from src.plotter import Plotter
from src.train_manager import TrainManager



@dataclass
class NetInfo:
    name: str
    object: Net
    train_loss: float = None
    test_loss: float = None


def main():
    # fine_tune_model()
    test_non_linearities()


def fine_tune_model():
    data_manager = DataManager()

    plotter = Plotter()

    train_manager = get_train_manager(data_manager)
    n_filters_list = []
    train_losses, test_losses = [], []
    for i in range(1, 11):
        n_filters = 2 * i
        net = Net(conv1_out_channels=n_filters)
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        train_manager.init_model(net, optimizer)
        train_and_save_model(train_manager)
        train_loss, test_loss = train_manager.get_losses()
        n_filters_list.append(n_filters)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    plotter.plot_filters_losses(n_filters_list, train_losses, test_losses)

    return train_losses, test_losses, n_filters_list


def test_non_linearities():
    print(f'testing non linearities...')
    non_linear_net = Net(fc1_out_channels=120, non_linear=True)
    linear_net = Net(fc1_out_channels=120, non_linear=False)
    big_linear_net = Net(fc1_out_channels=240, non_linear=False)

    nets = [NetInfo('non_linear', non_linear_net),
            NetInfo('linear', linear_net),
            NetInfo('big_linear', big_linear_net)]

    for net in nets:
        data_manager = DataManager()
        train_manager = get_train_manager(data_manager)

        # if net is not nets[1]: continue
        print(f'\ntesting {net.name}...')
        optimizer = optim.SGD(net.object.parameters(), lr=0.001, momentum=0.9)
        train_manager.init_model(net.object, optimizer)
        train_manager.train()
        train_loss, test_loss = train_manager.get_losses()
        print(f'train loss: {train_loss}\ntest loss: {test_loss}')


def get_train_manager(data_manager: DataManager):
    trainloader, testloader, classes = data_manager.load_cifar10()

    critertion = nn.CrossEntropyLoss()
    train_manager = TrainManager(trainloader, testloader, critertion)
    return train_manager


def train_and_save_model(train_manager: TrainManager):
    train_manager.train()
    train_manager.save_model()


def load_model(train_manager: TrainManager):
    train_manager.load_model()


def sandbox():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)


if __name__ == '__main__':
    main()
    # sandbox()
