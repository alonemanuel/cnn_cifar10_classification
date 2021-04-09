import os

import torch

from src.net import Net

N_EPOCHS = 2
MODEL_PATH = os.path.join('.', 'saved_models', 'cifar_net.pth')


class TrainManager:
    def __init__(self, trainloader, testloader, model, criterion, optimizer):
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self):
        for epoch in range(N_EPOCHS):
            running_loss = 0.0

            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                # print every 2000 mini-batches
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            print('finished training')

    def save_model(self):
        print(f'saving model...ux')
        torch.save(self.model.state_dict(), MODEL_PATH)

    def load_model(self):
        print(f'loading model...')
        self.model = Net()
        self.model.load_state_dict(torch.load(MODEL_PATH))

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
