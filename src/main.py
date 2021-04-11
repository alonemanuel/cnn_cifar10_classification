from dataclasses import dataclass

import numpy as np
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
    # test_non_linearities()
    test_locality_of_receptive_field()


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


def test_locality_of_receptive_field():
    print(f'testing locality of receptive field...')

    noshuffle_data_manager = DataManager(should_shuffle_images=False)
    shuffled_data_manager = DataManager(should_shuffle_images=True)

    for name, data_manager in [('no_shuffle', noshuffle_data_manager), ('shuffle', shuffled_data_manager)]:
        print(f'\ntesting {name}...')
        train_manager = get_train_manager(data_manager)
        net = Net()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        train_manager.init_model(net, optimizer)
        train_manager.train()
        print(f'\ntesting local field...')
        train_res, test_res = train_manager.get_losses()
        print(f'train loss: {train_res[0]}, train accuracy: {train_res[1]}\n'
              f'test loss: {test_res[0]}, test accuracy: {test_res[1]}')


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
    np_arr = np.array([[[[1, 2],
                         [3, 4]],
                        [[5, 6],
                         [7, 8]]],
                       [[[1, 3], [4, 6]], [[2, 5], [7, 9]]]])
    print(np_arr.shape)
    batch = torch.from_numpy(np_arr)
    print(batch)
    print(batch.size())
    batch = torch.reshape(batch, (2, 2, 4))
    print(batch.size())
    print(batch)

    r = torch.randperm(4)
    # c = torch.randperm(2)
    print()
    shuff_batch = batch[:, :, r]
    print(shuff_batch)
    # print(shuff_batch[:, :, 1])
    shuff_batch = torch.reshape(shuff_batch, (2, 2, 2, 2))
    print(shuff_batch)


if __name__ == '__main__':
    main()
    # sandbox()
