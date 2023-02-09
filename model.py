import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn.utils import prune
import collections
from visualization import ShowMatrix

train_dataset = datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
use_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define the model
class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x

class SparsityFFNN:
    def __init__(self):
        self.model = FFNN()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.model.to(device)
        self.criterion.to(device)
    
    def ResetModel(self):
        self.model = FFNN()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.model.to(device)
    
    def Train(self, epoches):
        for epoch in range(epoches):
            total_loss = 0
            for x, y in train_loader:
                self.optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print('Epoch:', epoch, 'Loss:', total_loss / len(train_loader))

    def Inference(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                total += y.size(0)
                output = self.model(x)
                _, predicted = torch.max(output.data, 1)
                temp = (predicted == y).sum().item()
                correct += temp
        print('Accuracy:', correct / total)

    def Prunning(self, sparsity_rate):
        parameters_to_prune = (
            (self.model.fc1, 'weight'),
            (self.model.fc1, 'bias'),
            (self.model.fc2, 'weight'),
            (self.model.fc2, 'bias'),
            (self.model.fc3, 'weight'),
            (self.model.fc3, 'bias'),
            (self.model.fc4, 'weight'),
            (self.model.fc4, 'bias'),
        )
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity_rate,
        )

    def CheckSparsity(self):
        print(
            "Sparsity in fc1.weight: {:.2f}%".format(
                100. * float(torch.sum(self.model.fc1.weight == 0))
                / float(self.model.fc1.weight.nelement())
            )
        )
        print(
            "Sparsity in fc1.bias: {:.2f}%".format(
                100. * float(torch.sum(self.model.fc1.bias == 0))
                / float(self.model.fc1.bias.nelement())
            )
        )
        print(
            "Sparsity in fc2.weight: {:.2f}%".format(
                100. * float(torch.sum(self.model.fc2.weight == 0))
                / float(self.model.fc2.weight.nelement())
            )
        )
        print(
            "Sparsity in fc2.bias: {:.2f}%".format(
                100. * float(torch.sum(self.model.fc2.bias == 0))
                / float(self.model.fc2.bias.nelement())
            )
        )
        print(
            "Sparsity in fc3.weight: {:.2f}%".format(
                100. * float(torch.sum(self.model.fc3.weight == 0))
                / float(self.model.fc3.weight.nelement())
            )
        )
        print(
            "Sparsity in fc3.bias: {:.2f}%".format(
                100. * float(torch.sum(self.model.fc3.bias == 0))
                / float(self.model.fc3.bias.nelement())
            )
        )
        print(
            "Sparsity in fc4.weight: {:.2f}%".format(
                100. * float(torch.sum(self.model.fc4.weight == 0))
                / float(self.model.fc4.weight.nelement())
            )
        )
        print(
            "Sparsity in fc4.bias: {:.2f}%".format(
                100. * float(torch.sum(self.model.fc4.bias == 0))
                / float(self.model.fc4.bias.nelement())
            )
        )
        print(
            "Global sparsity: {:.2f}%".format(
                100. * float(
                    torch.sum(self.model.fc1.weight == 0)
                    + torch.sum(self.model.fc1.bias == 0)
                    + torch.sum(self.model.fc2.weight == 0)
                    + torch.sum(self.model.fc2.bias == 0)
                    + torch.sum(self.model.fc3.weight == 0)
                    + torch.sum(self.model.fc3.bias == 0)
                    + torch.sum(self.model.fc4.weight == 0)
                    + torch.sum(self.model.fc4.bias == 0)
                )
                / float(
                    + self.model.fc1.weight.nelement()
                    + self.model.fc1.bias.nelement()
                    + self.model.fc2.weight.nelement()
                    + self.model.fc2.bias.nelement()
                    + self.model.fc3.weight.nelement()
                    + self.model.fc3.bias.nelement()
                    + self.model.fc4.weight.nelement()
                    + self.model.fc4.bias.nelement()
                )
            )
        )
    
    def DifferentPrunningRate(self, epoches, prunning_rate):
        print(f'////////////// Prunning Rate: {prunning_rate} //////////////')
        self.ResetModel()
        self.Train(epoches)
        self.Inference()
        self.Prunning(prunning_rate)
        self.CheckSparsity()
        self.Inference()
        self.Train(epoches)
        self.Inference()
        self.CheckSparsity()
    
    def ShowParameter(self, rate, type):
        fc1Weight = self.model.fc1.weight.to('cpu').detach().numpy()
        fc1Bias = self.model.fc1.bias.to('cpu').detach().numpy()
        fc2Weight = self.model.fc2.weight.to('cpu').detach().numpy()
        fc2Bias = self.model.fc2.bias.to('cpu').detach().numpy()
        fc3Weight = self.model.fc3.weight.to('cpu').detach().numpy()
        fc3Bias = self.model.fc3.bias.to('cpu').detach().numpy()
        fc4Weight = self.model.fc4.weight.to('cpu').detach().numpy()
        fc4Bias = self.model.fc4.bias.to('cpu').detach().numpy()
        ShowMatrix(fc1Weight, f'image/{type}/fc1_weight_{rate}.png')
        ShowMatrix(fc1Bias, f'image/{type}/fc1_bias_{rate}.png')
        ShowMatrix(fc2Weight, f'image/{type}/fc2_weight_{rate}.png')
        ShowMatrix(fc2Bias, f'image/{type}/fc2_bias_{rate}.png')
        ShowMatrix(fc3Weight, f'image/{type}/fc3_weight_{rate}.png')
        ShowMatrix(fc3Bias, f'image/{type}/fc3_bias_{rate}.png')
        ShowMatrix(fc4Weight, f'image/{type}/fc4_weight_{rate}.png')
        ShowMatrix(fc4Bias, f'image/{type}/fc4_bias_{rate}.png')
    
def main():
    sparsity = SparsityFFNN()
    #sparsity.DifferentPrunningRate(15, 0.7)
    sparsity.ShowParameter('80', 'original')
    sparsity.DifferentPrunningRate(15, 0.8)
    sparsity.ShowParameter('80', 'sparse')
    #sparsity.DifferentPrunningRate(15, 0.9)
    #sparsity.DifferentPrunningRate(15, 0.95)
    #sparsity.DifferentPrunningRate(15, 0.98)
    #sparsity.DifferentPrunningRate(15, 0.99)
    
if __name__ == "__main__":
    main()