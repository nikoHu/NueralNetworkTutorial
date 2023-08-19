import math
import matplotlib.pyplot as plt
import numpy as np

# 数据的真实输入输出值
X = [0.5, 1.0]
y = 0.8

# 网络的初始权值
w1 = 1.0
w2 = 0.5
w3 = 0.5
w4 = 0.7
w5 = 1.0
w6 = 2.0
w = [w1, w2, w3, w4, w5, w6]

# 网络的输出值，初始为0
y1 = 0

# 学习率
ita = 0.1

# 正向传播-根据输入计算输出
def foward(X, y, w):
    h1 = w[0] * X[0] + w[1] * X[1]
    h2 = w[2] * X[0] + w[3] * X[1]
    y1 = w[4] * h1 + w[5] + h2

    # print(y1, h1, h2)

    return y1, h1, h2

# 计算损失值
def loss(y, y1):
    # print(0.8 * math.pow(y - y1, 2))
    return 0.5 * math.pow(y - y1, 2)

# 反向传播 - 基于梯度下降更新网络权值
def backPropagation(loss, y1, h1, h2, w, X, ita):
    w_new = [0, 0, 0, 0, 0, 0]
    # 更新w4
    d_loss_div_d_y1 = 2 * 0.5 * (y - y1) * (-1)
    d_y1_div_d_w4 = h1 + 0
    d_loss_div_d_w4 = d_loss_div_d_y1 * d_y1_div_d_w4
    w_new[4] = w[4] - ita * d_loss_div_d_w4

    # 更新w5
    # d_loss_div_d_y1 = 2 * 0.5 * (y - y1) * (-1)
    d_y1_div_d_w5 = h2 + 0
    d_loss_div_d_w5 = d_loss_div_d_y1 * d_y1_div_d_w5
    w_new[5] = w[5] - ita * d_loss_div_d_w5

    # 更新w0
    d_y1_div_d_h1 = w[4] + 0
    d_h1_div_d_w0 = X[0]
    d_loss_div_d_w0 = d_loss_div_d_y1 * d_y1_div_d_h1 * d_h1_div_d_w0
    w_new[0] = w[0] - ita * d_loss_div_d_w0

    # 更新w1
    d_h1_div_d_w1 = X[1]
    d_loss_div_d_w1 = d_loss_div_d_y1 * d_y1_div_d_h1 * d_h1_div_d_w1
    w_new[1] = w[1] - ita * d_loss_div_d_w1 

    # 更新w2
    d_y1_div_d_h2 = w[5] + 0
    d_h2_div_d_w2 = X[0]
    d_loss_div_d_w2 = d_loss_div_d_y1 * d_y1_div_d_h2 * d_h2_div_d_w2
    w_new[2] = w[2] - ita * d_loss_div_d_w2

    # 更新w3
    d_h2_div_d_w3 = X[1]
    d_loss_div_d_w3 = d_loss_div_d_y1 * d_y1_div_d_h2 * d_h2_div_d_w3
    w_new[3] = w[3] - ita * d_loss_div_d_w3

    # print(w_new)
    return w_new

# 更新参数【实际网络中参数更新是在网络训练步骤完成的】
def train(w, X, y, ita):
    lossList = list()
    w_new = [0 for i in range(6)]
    w_new = w
    for i in range(20):
        y1, h1, h2 = foward(X, y, w_new)
        cur_loss = loss(y, y1)
        w_new = backPropagation(cur_loss, y1, h1, h2, w_new, X, ita)
        lossList.append(cur_loss)

    print(lossList)
    xAxis = [i for i in range(20)]
    lossList = np.array(lossList)
    xAxis = np.array(xAxis)

    plt.plot(xAxis, lossList)
    plt.xlabel("iteration number")
    plt.ylabel("loss")
    plt.show()

train(w, X, y, ita)