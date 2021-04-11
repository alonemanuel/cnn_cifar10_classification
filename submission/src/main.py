from dataclasses import dataclass
from random import shuffle

import numpy as np
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
    fine_tune_model()
    # test_non_linearities()
    # test_locality_of_receptive_field()
    # test_no_spatial_structure()


def fine_tune_model():
    print(f'\n\nfine tuning model...')
    data_manager = DataManager()

    plotter = Plotter()

    train_manager = get_train_manager(data_manager)
    n_filters_list = []
    train_losses, test_losses = [], []
    for i in range(1, 11):
        n_filters = 4 * i
        net = Net(conv1_out_channels=n_filters)
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        train_manager.init_model(net, optimizer)
        train_and_save_model(train_manager)
        (train_loss, train_accuracy), (test_loss, test_accuracy) = train_manager.get_losses()
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

    noshuffle_data_manager = DataManager(shuffle_type='none')
    shuffled_data_manager = DataManager(shuffle_type='fixed')

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


def test_no_spatial_structure():
    print(f'testing no spatial structure...')
    fixed_shuffle_data_manager = DataManager(shuffle_type='fixed')
    fresh_shuffle_data_manager = DataManager(shuffle_type='fresh')

    for name, data_manager in [('fixed_shuffle', fixed_shuffle_data_manager),
                               ('fresh_shuffle', fresh_shuffle_data_manager)]:
        print(f'\ntesting {name}...')
        train_manager = get_train_manager(data_manager)
        net = Net()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        train_manager.init_model(net, optimizer)
        train_manager.train()
        print(f'\ntesting no spatial...')
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
    print(f'sandboxing...')
    # n_filters = list(range(1,11))
    n_filters=[1,3,65,76,423,675,934,2345,13444,52345]
    # n_filters = shuffle(n_filters)
    n_filters = np.array(n_filters)
    loss_train = np.array(list(range(10))) * 2
    loss_test = np.array(list(range(10))) * 3
    plotter = Plotter()
    plotter.plot_filters_losses(n_filters, loss_train, loss_test)


if __name__ == '__main__':
    main()
    # sandbox()
