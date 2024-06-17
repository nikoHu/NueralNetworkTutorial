# 一开始优化器为optim.SGD 只训练1000轮，只能出现直线分割的效果；后面增加到10000轮训练，效果好多了。可见深度学习的调参是有多么的重要
# 目前调整到180000轮的时候，训练效果基本逼近网页效果
# 当优化器由optim.SGD 更改成 optim.Adam 之后，训练20000轮即可达到满意的效果
# 预测再调整学习率，网络层数等其他参数，能够在更少训练轮次时达到同等效果

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Generate the data 生成数据集
N = 100
D = 2
K = 3
X = np.zeros((N * K, D), dtype=np.float32)
y = np.zeros(N * K, dtype='int64')

def generate_data():
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)            # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2 # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

generate_data()

# Convert data to tensors
X = torch.from_numpy(X)
y = torch.from_numpy(y)

# Define the neural network model 定义网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(D, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, K)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model 实例化网络
model = SimpleNet()

# Define loss function and optimizer 定义损失函数 及 优化器[优化器定义采用何种方式的梯度下降，以及相应的学习率]
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.005) 
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training loop 开始训练
num_epochs = 80000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    # outputs = torch.sigmoid(outputs)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 画数据点的分解线（用等高线的方式）
def drawBoundary(data, net, label):
    h = 0.02
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # print(xx)
    
    row = len(xx)
    column = len(xx[0])
    xx = xx.reshape(row * column, 1)
    yy = yy.reshape(row * column, 1)
    inputData = np.c_[xx, yy]
    inputData_float32 = inputData.astype(np.float32)

    inputDataTensor = torch.from_numpy(inputData_float32)
    Z = []
    for x in inputDataTensor:
        # print(x)
        Z.append(net(x).tolist())
    #Z = np.array([net.predict(x) for x in inputData])
    Z = np.array(Z)
    Z = np.argmax(Z, axis = 1)
    Z = Z.reshape(len(Z), 1)
    # print(Z)
    
    Z = Z.reshape(row, column)
    xx = xx.reshape(row, column)
    yy = yy.reshape(row, column)
    # print(np.shape(Z))
    
    fig = plt.figure()
    plt.contourf(xx, yy, Z)# , cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(data[:, 0], data[:, 1], c = label, s = 10, cmap = plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    #fig.savefig('spiral_net.png')
    plt.show()

drawBoundary(X, model, y)