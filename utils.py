import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, net, trainloader, testloader, device_type, archi_type=None):
        self.device = torch.device(device_type)
        self.net = net.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.archi_type = archi_type
    
    def test(self):
        self.net.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)

                # forward pass
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total
    
    def train(self, epochs=10):
        if self.archi_type == 'lenet':
            optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        else:
            optimizer = optim.Adam(self.net.parameters(), lr=0.01)
        cost = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.net.train()
            for images, labels in self.trainloader:  
                images, labels = images.to(self.device), labels.to(self.device)

                # forward pass
                outputs = self.net(images)
                loss = cost(outputs, labels)
                
                # backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print("epoch:", epoch, ",test accuracy:", self.test())


