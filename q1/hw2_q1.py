# -*- coding: utf-8 -*-


#https://github.com/MedMNIST/MedMNIST


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms

from medmnist import BloodMNIST, INFO

import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score


device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64

# Data Loading

data_flag = 'bloodmnist'
print(data_flag)
info = INFO[data_flag]
print(len(info['label']))
n_classes = len(info['label'])

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

import time

# --------- Before Training ----------
total_start = time.time()

#Training Function

def train_epoch(loader, model, criterion, optimizer):
    
    ### YOUR CODE HERE ###
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    return total_loss / len(loader)

#Evaluation Function

def evaluate(loader, model, apply_softmax=False):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.squeeze().long()

            outputs = model(imgs, apply_softmax=apply_softmax)
            preds += outputs.argmax(dim=1).cpu().tolist()
            targets += labels.tolist()

    return accuracy_score(targets, preds)


def plot(epochs, plottable, ylabel='', name=''):
    print(plottable)
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(range(epochs), plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')

train_dataset = BloodMNIST(split='train', transform=transform, download=True, size=28)
val_dataset   = BloodMNIST(split='val',   transform=transform, download=True, size=28)
test_dataset  = BloodMNIST(split='test',  transform=transform, download=True, size=28)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# initialize the model
# get an optimizer
# get a loss criterion

### YOUR CODE HERE ###
class CNNModel(nn.Module):
    def __init__(self, in_channels, num_classes, input_hw: tuple[int, int]):
        super().__init__()
        H, W = input_hw
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate correct dimensions after pooling
        # Input: (H, W) = (28, 28)
        # After conv1 + pool: (14, 14)
        # After conv2 + pool: (7, 7)  
        # After conv3 + pool: (3, 3)
        conv_output_size = 128 * (H // 8) * (W // 8)  # 128 channels, spatial dims reduced by 8 (2^3 from 3 pools)

        # --- Fully connected layers ---
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        

    def forward(self, x, apply_softmax=False):
        x = self.pool(self.relu(self.conv1(x)))

        # Block 2
        x = self.pool(self.relu(self.conv2(x)))
       

        # Block 3
        x = self.pool(self.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)  # (N, 128*3*3) = (N, 1152)

        # FCs
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)

        if apply_softmax:
            return F.softmax(logits, dim=1)  # probabilities
        return logits  # raw scores (logits)

# training loop
### you can use the code below or implement your own loop ###

model = CNNModel(in_channels=3, num_classes=n_classes, input_hw=(28, 28)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 200
train_losses = []
val_accs = []
test_accs = []
softmax_accs = []
epochs_without_improvement = 0
best_val_acc = 0.0
for epoch in range(epochs):

    epoch_start = time.time()

    train_loss = train_epoch(train_loader, model, criterion, optimizer)
    val_acc = evaluate(val_loader, model, apply_softmax=True)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} ")

    train_losses.append(train_loss)
    val_accs.append(val_acc)

    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | "
          f"Time: {epoch_time:.2f} sec")
    
    #Test Accuracy
    test_acc = evaluate(test_loader, model)
    print("Test Accuracy:", test_acc)
    test_accs.append(test_acc)
    
    # Early Stopping
    # if val_acc > best_val_acc:
    #     best_val_acc = val_acc
    #     epochs_without_improvement = 0
    # else:
    #     epochs_without_improvement += 1

    # if epochs_without_improvement >= 5:
    #     print("Early stopping triggered.")
    #     break




#Save the model
torch.save(model.state_dict(), "bloodmnist_cnn-with-softmax-and-pooling.pth")
print("Model saved as bloodmnist_cnn.pth")


# --------- After Training ----------
total_end = time.time()
total_time = total_end - total_start

print(f"\nTotal training time: {total_time/60:.2f} minutes "
      f"({total_time:.2f} seconds)")

#print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))

#config = "{}-{}-{}-{}-{}".format(opt.learning_rate, opt.optimizer, opt.no_maxpool, opt.no_softmax,)
config = "{}".format(str(0.1))


plot(epochs, train_losses, ylabel='Loss', name='CNN-training-loss-with-softmax-and-pooling{}'.format(config))
plot(epochs, val_accs, ylabel='Accuracy', name='CNN-validation-accuracy-with-softmax-and-pooling-{}'.format(config))
plot(epochs, test_accs, ylabel='Accuracy', name='CNN-test-accuracy-with-softmax-and-pooling-{}'.format(config))