import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

N_PIXELS = 1024


class DataManager:

    def __init__(self, should_shuffle_images=False):
        self.should_shuffle_images = should_shuffle_images
        self.r = torch.randperm(N_PIXELS)

    def load_cifar10(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             transforms.Lambda(self.shuffle_images if self.should_shuffle_images else identity)])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return trainloader, testloader, classes

    def shuffle_images(self, x):
        # print(f'x: {x}')
        # print(f'x size: {x.size()}')
        x = torch.reshape(x, (3, -1))
        # print(f'x reshaped: {x}')
        # print(f'x reshaped size: {x.size()}')
        n = x.size()[-1]
        # print(f'n: {n}')
        shuffled_x = x[:, self.r]
        # print(f'shuffled x: {shuffled_x}')
        x = torch.reshape(shuffled_x, (3, 32, 32))
        return x


def identity(x):
    return x
