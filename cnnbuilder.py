import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


class CNNBuilder:
    """
    Attributes:
        trainloader
        testloader
        classes
    """
    def __init__(self, net, device_type, archi_type, dataset):
        self.device = torch.device(device_type)
        self.net = net.to(self.device)
        self.archi_type = archi_type
        self.dataset = dataset
    
    def load_dataset(self, data_dir):
        # prepare dataset

        # list image transformations to perform
        transform = transforms.Compose([
            transforms.ToTensor(),  # transform to tensor
            transforms.Normalize((0.5,), (0.5,))  # scale to [-1, 1]
        ])

        # load dataset (available in torch)

        if self.dataset == 'fashionmnist':
            trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
            testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
            self.classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            self.classes_short = self.classes
        elif self.dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
            self.classes = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 
                            'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
            self.classes_short = self.classes
        elif self.dataset == 'eurosat':
            entireset = torchvision.datasets.EuroSAT(root=data_dir, download=True, transform=transform)
            trainset, testset = torch.utils.data.random_split(entireset, [24300, 2700])
            self.classes = ['Annual Crop', 'Forest', 'Herbaceous Vegetation', 'Highway', 'Industrial', 
                            'Pasture', 'Permanent Crop', 'Residential', 'River', 'Sea/Lake']
            self.classes_short = ['Ann. Crop', 'Forest', 'H. Vegetation', 'Highway', 'Industrial', 
                            'Pasture', 'Perm. Crop', 'Residential', 'River', 'Sea/Lake']

        # save as attribute
        self.testset = testset

        # use dataloader to load and iterate on dataset easily
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)
    
    def train(self, epochs=10, save_path='models/{}_{}.pth', verbose=True):
        if self.archi_type == 'lenet':
            optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        else:
            optimizer = optim.Adam(self.net.parameters(), lr=0.01)
        cost = nn.CrossEntropyLoss()

        # save historical losses and accs
        hist_metrics = dict()
        hist_metrics['epoch'] = []
        hist_metrics['train_loss'] = []
        hist_metrics['train_acc'] = []
        hist_metrics['test_loss'] = []
        hist_metrics['test_acc'] = []

        for epoch in range(epochs):
            self.net.train()
            correct = 0
            total = 0
            total_loss = 0

            for images, labels in self.trainloader:  
                images, labels = images.to(self.device), labels.to(self.device)

                # forward pass
                outputs = self.net(images)
                loss = cost(outputs, labels)

                # compute metrics: loss and acc
                total_loss += (loss.item() * labels.size(0))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            train_loss, train_acc = total_loss / total, correct / total
            test_loss, test_acc = self.compute_metrics()

            hist_metrics['epoch'].append(epoch)
            hist_metrics['train_loss'].append(train_loss)
            hist_metrics['train_acc'].append(train_acc)
            hist_metrics['test_loss'].append(test_loss)
            hist_metrics['test_acc'].append(test_acc)

            if verbose:
                print("epoch: {0}, train loss {1:.2f}, train acc: {2:.2f}, test loss: {3:.2f}, test acc: {4:.2f}"
                      .format(epoch, train_loss, train_acc, test_loss, test_acc))
        
        save_path = save_path.format(self.dataset, self.archi_type)
        torch.save(self.net.state_dict(), save_path)
        print('Model saved to: {}'.format(save_path))
        return hist_metrics
    
    def load_model(self, save_path='models/{}_{}.pth'):
        save_path = save_path.format(self.dataset, self.archi_type)
        self.net.load_state_dict(torch.load(save_path, map_location=lambda storage, loc: storage))
        print('Model loaded from: {}'.format(save_path))
        self.net.to(self.device)
        self.net.eval()

    def compute_metrics(self, save_predictions=False, train=False):
        self.net.eval()
        correct = 0
        total = 0
        losses = []
        predictions = []

        if train:
            dataloader = self.trainloader
        else:
            dataloader = self.testloader

        cost = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                # forward pass
                outputs = self.net(images)
                loss = cost(outputs, labels)

                # compute metrics: loss and acc
                losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                if save_predictions:
                    predictions.append(predicted)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if save_predictions:
            self.predictions = torch.cat(predictions)
        return np.array(losses).mean(), correct / total

    def plot_metrics(self, metrics, save_path='logs/{}/{}.{}'):
        plt.plot(metrics['train_acc'])
        plt.plot(metrics['test_acc'])
        plt.ylabel("accuracy")
        plt.xlabel("epochs")
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(save_path.format(self.dataset, self.archi_type + '_acc', "png"))
        plt.show()

        plt.plot(metrics['train_loss'])
        plt.plot(metrics['test_loss'])
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(save_path.format(self.dataset, self.archi_type + '_loss', "png"))
        plt.show()

        with open(save_path.format(self.dataset, self.archi_type, "json"), 'w') as f:
            json.dump(metrics, f)


    def generate_classification_report(self, save_path='logs/{}/{}.{}'):
        tmp_testset = torch.utils.data.DataLoader(self.testset, batch_size=len(self.testset), shuffle=False)
        self.net.eval()
        with torch.no_grad():
            for _, labels in tmp_testset:
                report = classification_report(labels.cpu().numpy(), self.predictions.cpu().numpy(), output_dict=True)
                with open(save_path.format(self.dataset, self.archi_type, "json"), 'w') as f:
                    json.dump(report, f)

    def plot_images(self, n_samples=10, save_path='logs/{}/{}_images.png'):
        device = torch.device('cpu')
        tmp_testset = torch.utils.data.DataLoader(self.testset, batch_size=len(self.testset), shuffle=False)
        
        fig, axs = plt.subplots(len(self.classes), n_samples)
        fig.set_size_inches(13, 11)
        fig.set_dpi(32)
        text_style_title = dict(ha='right', va='center', fontsize=14, fontfamily='monospace')
        text_style_mis = dict(ha='center', va='bottom', fontsize=10, fontfamily='monospace', color='red')

        for images, labels in tmp_testset:
            images, labels = images.to(device), labels.to(device)
            images = (images + 1) * 0.5
            for c in range(len(self.classes)):
                random_i = np.random.choice(np.array(self.predictions.cpu().numpy() == c).nonzero()[0], 10)
                col = 0
                for i in random_i:
                    if col == 0:
                        text_style_title['transform'] = axs[c, col].transAxes
                        axs[c, col].text(-0.1, 0.5, self.classes[c], **text_style_title)
                    if images[i].shape[0] == 1:
                        axs[c, col].matshow(images[i][:, :, ].permute(1, 2, 0).cpu().numpy(), cmap=plt.cm.Greys)
                    else:
                        axs[c, col].matshow(images[i][:, :, ].permute(1, 2, 0).cpu().numpy())
                    axs[c, col].set_xticks([])
                    axs[c, col].set_yticks([])
                    axs[c, col].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                    if labels[i] != c:
                        text_style_mis['transform'] = axs[c, col].transAxes
                        plt.setp(axs[c, col].spines.values(), color='red')
                        axs[c, col].text(0.5, 1, self.classes_short[labels[i]], **text_style_mis)
                    col += 1
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0.25)
        plt.savefig(save_path.format(self.dataset, self.archi_type))
        plt.show()

