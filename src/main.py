import torch.nn as nn
import torch.optim as optim

from src.data_manager import DataManager
from src.net import Net
from src.train_manager import TrainManager

N_EPOCHS = 2


def main():
    # example_rand_input()
    data_manager = DataManager()
    train_manager = get_train_manager(data_manager)
    train_manager.load_model()
    train_manager.test_model()
    # loss = get_loss(trainloader)


def get_train_manager(data_manager: DataManager):
    trainloader, testloader, classes = data_manager.load_cifar10()
    net = Net()
    critertion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train_manager = TrainManager(trainloader, testloader, net, critertion, optimizer)
    return train_manager


def train_and_save_model(train_manager: TrainManager):
    train_manager.train()
    train_manager.save_model()


def load_model(train_manager: TrainManager):
    train_manager.load_model()


if __name__ == '__main__':
    main()
