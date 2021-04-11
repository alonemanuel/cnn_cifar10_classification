import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, conv1_out_channels=20, fc1_out_channels=120, non_linear=True):
        super(Net, self).__init__()
        print(
            f'building {"non_linear" if non_linear else "linear"} net with {conv1_out_channels} conv1 and {fc1_out_channels} fc1')
        # 3 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, conv1_out_channels, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv1_out_channels, 16, (5, 5))
        # an affine operation: y=Wx + b
        # todo: why 120? why 84? just random?
        self.fc1 = nn.Linear(16 * 5 * 5, fc1_out_channels)
        self.fc2 = nn.Linear(fc1_out_channels, 84)
        self.fc3 = nn.Linear(84, 10)

        self.net_name = f'conv1_{conv1_out_channels}_fc1_{fc1_out_channels}'
        self.non_linear = non_linear

    def get_num_filters(self):
        print(f'in get_num_of_filters types...')

        n_filters = self.conv1.out_channels
        n_filters += self.conv2.out_channels
        # what's condsidered a neuron?
        # n_filters += self.fc1x
        return n_filters

    def forward(self, x):
        # relu if has non-linearities, identity otherwise
        activation = F.relu if self.non_linear else lambda obj: obj

        # Max pooling over a (2, 2) window
        x = F.max_pool2d(activation(self.conv1(x)), (2, 2))
        # if the size is a square you can only specify a single number
        x = F.max_pool2d(activation(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = activation(self.fc1(x))
        x = activation(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # all dimensions except the batch dimension
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
