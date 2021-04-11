import os

import torch
from tqdm import tqdm

from src.net import Net

N_EPOCHS = 2
MODEL_DIR = os.path.join('.', 'saved_models')
USE_CUDA = True


class TrainManager:
    def __init__(self, trainloader, testloader, criterion):
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.train_losses_by_epoch = []
        # todo: after each epoch do we test on all test set?
        self.test_losses_by_epoch = []
        if USE_CUDA:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        print(f'device is: {self.device}')

    def train(self):
        print(f'training model...')
        for epoch in tqdm(range(N_EPOCHS), position=0, leave=True):
            epoch_running_loss = 0.0
            running_loss = 0.0
            for i, data in enumerate(tqdm(self.trainloader, position=0, leave=True), 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = [data[i].to(self.device) for i in [0, 1]]

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                epoch_running_loss += loss.item()
                # print every 2000 mini-batches
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            self.train_losses_by_epoch.append(epoch_running_loss / len(self.trainloader))
        print('finished training')

    @staticmethod
    def get_model_path(model_name):
        model_path = os.path.join(MODEL_DIR, f'cifar_net_{model_name}.pth')
        return model_path

    def save_model(self):
        print(f'saving model...')
        torch.save(self.model.state_dict(), self.get_model_path(self.model.net_name))

    @staticmethod
    def get_saved_model(model_name):
        print(f'loading model...')
        model = Net()
        model.load_state_dict(torch.load(TrainManager.get_model_path(model_name)))
        return model

    def test_model(self):
        print(f'testing model...')
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    def get_loss(self, data_loader):
        print(f'getting loss...')
        # todo: is the test loss just the average over all batches?
        running_loss = 0.0
        for i, data in enumerate(tqdm(data_loader, leave=True, position=0), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = [data[i].to(self.device) for i in [0, 1]]
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            running_loss += loss.item()

        n_batches = len(data_loader)
        avg_loss = running_loss / n_batches
        print(f'loss is {avg_loss}')
        return avg_loss

    def get_losses(self):
        train_loss = self.get_loss(self.trainloader)
        test_loss = self.get_loss(self.testloader)
        return train_loss, test_loss

    def init_model(self, model: Net, optimizer):
        self.model = model
        self.model.to(self.device)
        self.optimizer = optimizer
