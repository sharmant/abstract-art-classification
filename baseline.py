import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=0) # 0 = baseline
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--dev_every', type=int, default=2)
parser.add_argument('--data_input', type=str, default="train.p,dev.p,test.p") # comma separated pkl files 
parser.add_argument('--checkpoint_dir', type=str, default="checkpoints")
parser.add_argument('--suffix', type=str, default="")

args = parser.parse_args()

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1) 
        self.pool = nn.MaxPool2d(2,2)              
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.dropout = nn.Dropout(args.dropout)              
        self.fc1 = nn.Linear(32*32*64, 256)        
        self.fc2 = nn.Linear(256, 84)              
        self.fc3 = nn.Linear(84, 2)               
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 256x256x8
        x = self.pool(F.relu(self.conv2(x))) # 128x16
        x = self.pool(F.relu(self.conv3(x))) # 64x32
        x = self.pool(F.relu(self.conv4(x))) # 32x64
        x = self.dropout(x)   
        x = x.view(-1, 32*32*64)
        x = F.relu(self.fc1(x))              # 256
        x = self.dropout(F.relu(self.fc2(x)))# 84
        x = self.softmax(self.fc3(x))        # 2
        return x

if args.model == 0:
    model = VanillaCNN()
    print("=> Initializing baseline model")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device is {}".format(device))
model = model.to(device)
print("Models loaded on device")

(X_train_0, y_train_0) = pickle.load(open("train_da_0.p", "rb"))
(X_train_1, y_train_1) = pickle.load(open("train_da_1.p", "rb"))
(X_dev, y_dev) = pickle.load(open("dev.p", "rb"))
(X_test, y_test) = pickle.load(open("test.p", "rb"))

X_train_0, y_train_0 = torch.Tensor(X_train_0).permute(0,3,1,2), torch.Tensor(y_train_0).long()
X_train_1, y_train_1 = torch.Tensor(X_train_1).permute(0,3,1,2), torch.Tensor(y_train_1).long()

X_dev, y_dev = torch.Tensor(X_dev).permute(0,3,1,2), torch.Tensor(y_dev).long()
X_test, y_test = torch.Tensor(X_test).permute(0,3,1,2), torch.Tensor(y_test).long()

X_train_0 = X_train_0.to(device)
X_train_1 = X_train_1.to(device)
X_dev = X_dev.to(device)
X_test = X_test.to(device)

y_train_0 = y_train_0.to(device)
y_train_1 = y_train_1.to(device)
y_dev = y_dev.to(device)
y_test = y_test.to(device)

criterion = torch.nn.CrossEntropyLoss()# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = criterion.to(device)

# Train the model
losses = []
accuracies = []
dev_losses = []
dev_accuracies = []

print("=> Training")
X_batches = [X_train_0[:71], X_train_0[71:141], X_train_0[141:211], X_train_0[211:281], X_train_0[281:351], X_train_0[351:421], X_train_0[421:491], X_train_0[491:], X_train_1[:71], X_train_1[71:141], X_train_1[141:211], X_train_1[211:281], X_train_1[281:351], X_train_1[351:421], X_train_1[421:491], X_train_1[491:]]

y_batches = [y_train_0[:71], y_train_0[71:141], y_train_0[141:211], y_train_0[211:281], y_train_0[281:351], y_train_0[351:421], y_train_0[421:491], y_train_0[491:], y_train_1[:71], y_train_1[71:141], y_train_1[141:211], y_train_1[211:281], y_train_1[281:351], y_train_1[351:421], y_train_1[421:491], y_train_1[491:]]

for epoch in range(1, args.num_epochs+1):
    loss_sum = 0
    accuracy_sum = 0
    for b in range(len(X_batches)):
        X_b, y_b = X_batches[b], y_batches[b]

        # forward
        # print("forward")
        outputs = model(X_b)
        # print("outputs: {}".format(outputs.shape))
        loss = criterion(outputs, y_b)
        loss_sum += loss.item()
        # losses.append(loss.item())

        # backprop
        # print("backprop")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print("eval")
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y_b).sum().item()
        accuracy = float(correct)
        accuracy_sum += accuracy
        # accuracies.append(accuracy)

    loss_avg = loss_sum / (X_train_0.shape[0] + X_train_1.shape[0])
    accuracy_avg = accuracy_sum / (X_train_0.shape[0] + X_train_1.shape[0])
    losses.append(loss_avg)
    accuracies.append(accuracy_avg)
    if epoch % 10 == 0:
        print("Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%".format(epoch, args.num_epochs, loss_avg, accuracy_avg*100))
    
    if epoch % args.dev_every == 0:
        model.eval()
        with torch.no_grad():
            dev_outputs = model(X_dev)

            dev_loss = criterion(dev_outputs, y_dev)
            dev_losses.append(dev_loss.item())

            _, dev_predicted = torch.max(dev_outputs, 1)
            dev_correct = (dev_predicted == y_dev).sum().item()
            dev_accuracy = float(dev_correct) / X_dev.shape[0]
            dev_accuracies.append(dev_accuracy)
            print('Dev Accuracy: {}%'.format(dev_accuracy * 100))

    # print("losses: {}".format(losses))
    # print("accuracies: {}".format(accuracies))
    # print("dev losses: {}".format(dev_losses))
    # print("dev accuracies: {}".format(dev_accuracies))

# Evaluate the model
print("=> Evaluating")
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, test_predicted = torch.max(test_outputs, 1)
    test_correct = (test_predicted == y_test).sum().item()
    test_accuracy = float(test_correct) / X_test.shape[0]
    print('Test Accuracy: {}%'.format(test_accuracy * 100))

with torch.no_grad():
    dev_outputs = model(X_dev)
    _, dev_predicted = torch.max(dev_outputs, 1)
    dev_correct = (dev_predicted == y_dev).sum().item()
    dev_accuracy = float(dev_correct) / X_dev.shape[0]
    print('Dev Accuracy: {}%'.format(dev_accuracy * 100))

# Save the model and plot
print("=> Saving checkpoint")
torch.save(model.state_dict(), args.checkpoint_dir + '/baseline_{}.ckpt'.format(args.num_epochs))

# Plot train/dev accuracy curves
print("Plotting losses and accuracies")
acc_plot = plt.figure()
plt.title = "Train vs. Dev Accuracy"
plt.xlabel = "Number of epochs"
plt.ylabel = "Accuracy"
plt.plot(
    range(1,args.num_epochs+1), accuracies,
    [i for i in range(1,args.num_epochs+1) if i % args.dev_every == 0], dev_accuracies
)
plt.savefig("baseline_train_dev_acc_{0}_{1}.png".format(args.num_epochs, args.suffix))

loss_plot = plt.figure()
plt.title = "Train vs. Dev Loss"
plt.xlabel = "Number of epochs"
plt.ylabel = "Loss"
plt.plot(
    range(1,args.num_epochs+1), losses,
    [i for i in range(1,args.num_epochs+1) if i % args.dev_every == 0], dev_accuracies
)
plt.savefig("baseline_train_dev_loss_{0}_{1}.png".format(args.num_epochs, args.suffix))

# ax1 = train_loss.add_subplot(111)
# ax1.plot()
# ax1 = f1.add_subplot(111)
# ax1.plot(range(0,10))
# ax2 = f2.add_subplot(111)
# ax2.plot(range(10,20))
# plt.show()