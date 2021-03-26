import torch.nn as nn
import torch.optim as optim

from src.data_manager import DataManager
from src.net import Net
from src.train_manager import TrainManager
from src.utils import show_example_images

N_EPOCHS = 2


def main():
    # example_rand_input()
    data_manager = DataManager()
    trainloader, testloader, classes = data_manager.load_cifar10()
    show_example_images(trainloader, classes)
    net = Net()
    critertion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train_manager = TrainManager(trainloader, testloader, net, critertion, optimizer)
    train_manager.train()


if __name__ == '__main__':
    main()
