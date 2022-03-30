import datetime
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from keras.datasets import mnist
from keras import backend as K

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
learning_rate = 0.001
batch_size = 100
epochs = 1
dropout = 0.25
units = 128
num_steps = 60000 / 100
img_rows, img_cols = 28, 28
num_classes = 10

type = 1

# --------------------------------------------------
# 学習用に60,000個、検証用に10,000個
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 60000x28x28
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
# x_train -= 0.5
x_test /= 255
# x_test -= 0.5
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# --------------------------------------------------

if type == 1:
    # --------------------------------------------------
    print(x_train[0][5])
    # print(x_train[0][0].size())
    x_train = x_train.reshape([60000 * 784])
    x_train = x_train.reshape([60000, 1, 28, 28])
    # print(x_train[0][0][5])

    x = torch.tensor(x_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.int64)
    train_dataset = torch.utils.data.TensorDataset(x, y)
    print(train_dataset[0][0].size())
    print(train_dataset[0][1])
    # --------------------------------------------------

    # --------------------------------------------------
    x_test = x_test.reshape([10000 * 784])
    x_test = x_test.reshape([10000, 1, 28, 28])

    x2 = torch.tensor(x_test, dtype=torch.float32)
    y2 = torch.tensor(y_test, dtype=torch.int64)
    test_dataset = torch.utils.data.TensorDataset(x2, y2)
    print(test_dataset[0][0].size())
    print(test_dataset[0][1])
    # --------------------------------------------------
else:

    # for x,y in train_dataset:
    #      print(x)
    #      print(y)

    # print(train_dataset[1][0])
    # for x,y in train_dataset:
    #     print(x,y)
    # --------------------------------------------------
    # MNIST dataset
    # 60000x1x28x28
    # 60000
    trans = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (1.0, ))])
    train_dataset = torchvision.datasets.MNIST(root='./data/',
                                               train=True,
                                               transform=trans,
                                               download=True)
    # 60000x1x28x28
    # for x,y in train_dataset:
    #     print(x,y)
    print(train_dataset[0][0].size())
    print(train_dataset[0][1])

    test_dataset = torchvision.datasets.MNIST(root='./data/',
                                              train=False,
                                              transform=trans)
    # for x,y in test_dataset:
    #      print(x)
    #      print(y)
    #      print(x.size(),y)

    print(test_dataset[0][0].size())
    print(test_dataset[1][0].size())
    # --------------------------------------------------

# --------------------------------------------------
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# for i in train_loader:
#     print(i)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Convolutional neural network (two convolutional layers)


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                      stride=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # self.fc = nn.Linear(7*7*32, num_classes)
        self.fc = nn.Linear(64 * 5 * 5, 10)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        return out


model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
print('------------------- train start:', datetime.datetime.today())
total_step = len(train_loader)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # print(images.size())
        # print(labels.size())

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch + 1, epochs, i + 1, total_step, loss.item()))

print('------------------- train end:', datetime.datetime.today())
# Test the model
print('------------------- test start:', datetime.datetime.today())
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(
        100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
