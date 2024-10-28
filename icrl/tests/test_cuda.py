import time
import torch
from torch import nn
import numpy as np

w0 = np.array([[2.0, -3.0]])
b0 = np.array([[10.0]])
# 定义模型
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn_like(torch.Tensor(w0)))
        self.b = nn.Parameter(torch.zeros_like(torch.Tensor(b0)))
    #正向传播
    def forward(self,x):
        return x@self.w.t() + self.b

linear = LinearRegression()

# 训练模型
optimizer = torch.optim.Adam(linear.parameters(),lr = 0.1)
loss_func = nn.MSELoss()



def train(epoches):
    tic = time.time()
    for epoch in range(epoches):
        n = 100  # 样本数量

        X = 10 * np.random.rand(n, 2) - 5.0  # torch.rand是均匀分布

        Y = X @ w0.T + b0 + np.random.normal(0.0, 2.0, size=[n, 1])  # @表示矩阵乘法,增加正态扰动
        X = torch.Tensor(X, )
        Y = torch.Tensor(Y, )
        optimizer.zero_grad()
        Y_pred = linear(X)
        loss = loss_func(Y_pred,Y)
        loss.backward()
        optimizer.step()
        if epoch%50==0:
            print({"epoch":epoch,"loss":loss.item()})
    toc = time.time()
    print("time used:",toc-tic)

train(500)

# 定义模型
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn_like(torch.Tensor(w0)))
        self.b = nn.Parameter(torch.zeros_like(torch.Tensor(b0)))
    #正向传播
    def forward(self,x):
        return x@self.w.t() + self.b

linear = LinearRegression()

# 移动模型到GPU上
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
linear.to(device)

#查看模型是否已经移动到GPU上
print("if on cuda:",next(linear.parameters()).is_cuda)


# 训练模型
optimizer = torch.optim.Adam(linear.parameters(),lr = 0.1)

def train(epoches):
    tic = time.time()
    for epoch in range(epoches):
        # 准备数据
        loss_func = nn.MSELoss()
        n = 100  # 样本数量

        X = 10 * np.random.rand(n, 2) - 5.0  # torch.rand是均匀分布

        Y = X @ w0.T + b0 + np.random.normal(0.0, 2.0, size=[n, 1])  # @表示矩阵乘法,增加正态扰动

        # 移动到GPU上
        # print("torch.cuda.is_available() = ", torch.cuda.is_available())
        X = torch.tensor(X, device=device, dtype=torch.float32)
        Y = torch.tensor(Y, device=device, dtype=torch.float32)
        # print("X.device:", X.device)
        # print("Y.device:", Y.device)
        optimizer.zero_grad()
        Y_pred = linear(X)
        loss = loss_func(Y_pred,Y)
        loss.backward()
        optimizer.step()
        if epoch%50==0:
            print({"epoch":epoch,"loss":loss.item()})
    toc = time.time()
    print("time used:",toc-tic)

train(500)
