import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


#初始化生成高斯随机分布
def init_data():
    #中心
    mean_a = [0, 0]
    mean_b = [1, 2]
    #协方差矩阵
    cov_a = [[1, 0], [0, 1]]
    cov_b = [[1, 0], [0, 2]]
    #高斯分布
    point_a = np.random.multivariate_normal(mean_a, cov_a, 300)
    point_b = np.random.multivariate_normal(mean_b, cov_b, 200)
    #按行的方式合并
    data = np.append(point_a, point_b, 0)
    #设置标签
    labels = [0] * 500
    labels = np.array(labels)
    #A类
    labels[0:300] = 0
    #B类
    labels[300:] = 1
    return np.round(data, 3), labels


#dataSet为坐标值,labels为类别
dataSet, labels = init_data()
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.

    #前向传播
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        z = sigmoid(z)
        return z

    #计算损失,用的是均方差损失函数
    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost

    #求得反向梯度
    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z - y) * x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)
        return gradient_w, gradient_b

    #更新梯度数值
    def update(self, gradient_w, gradient_b, eta=0.1):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b

    def train(self, x, y, num_epoches, batch_size=50, eta=0.01):
        n = len(x)
        losses = []
        for epoch_id in range(num_epoches):
            # 在每轮迭代开始之前，将训练数据的顺序随机打乱
            # 然后再按每次取batch_size条数据的方式取出
            shuffleIndex = np.random.permutation(np.arange(len(x)))
            x = x[shuffleIndex]
            y = y[shuffleIndex]
            # 将训练数据进行拆分，每个mini_batch包含batch_size条的数据
            mini_batches_x = [
                x[k:k + batch_size] for k in range(0, n, batch_size)
            ]
            mini_batches_y = [
                y[k:k + batch_size] for k in range(0, n, batch_size)
            ]
            for iter_id in range(len(mini_batches_x)):

                feature = mini_batches_x[iter_id]
                label = mini_batches_y[iter_id]
                a = self.forward(feature)
                loss = self.loss(a, label)
                gradient_w, gradient_b = self.gradient(feature, label)
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
        return losses, self.w, self.b
#用于计算预测值的函数
def predict(input_point, weight, bias):
    z = np.dot(input_point, weight) + bias
    z = sigmoid(z)
    return z
accuracy = []
epoch_num = []
loss = []
for epoch in range(1, 1000, 50):
    net = Network(2)
    labels = labels.reshape(500, 1)
    #启动训练
    losses, weight, bias = net.train(dataSet, labels, epoch, eta=0.01)
    # print(losses)
    #计算准确率
    correct = 0
    for i in range(500):
        if (i < 300):
            if (predict(dataSet[i], weight, bias) < 0.5):
                correct += 1
        else:
            if (predict(dataSet[i], weight, bias) > 0.5):
                correct += 1
    accuracy.append(correct / 500)
    epoch_num.append(epoch)
# 画出损失函数的变化趋势
plt.figure()
plt.title("不同epoch数值下逻辑回归的正确率")
plt.plot(epoch_num, accuracy, marker='o')

best_epoch_index = np.array(accuracy).argmax()
best_epoch = epoch_num[best_epoch_index]
net = Network(2)
losses, weight, bias = net.train(dataSet, labels, best_epoch, eta=0.01)

plt.figure()
plt.title("最优epoch下训练时loss的变化趋势")
plt.plot(losses)

# 绘制生成的a,b点分布图
plt.figure()
x1, y1 = dataSet[0:300].T
x2, y2 = dataSet[300:500].T
plt.scatter(x1, y1, c='y', marker='o', alpha=0.5)
plt.scatter(x2, y2, c='r', marker='o', alpha=0.5)
plt.title("A,B两类点的分布图")
plt.xlabel("x")
plt.ylabel("y")
x = np.arange(-3.0, 4.0, 0.1)  # 绘制的线的显示范围，最小x，最大x
y = (-bias - weight[0] * x) / weight[1]  # 绘制的函数方程
plt.plot(x, y)
plt.show()