N_EPOCHS = 2


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
                if i%2000 ==1999:
                    print('[')