import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# from pylab import *

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
    labels[0:300] = 1
    #B类
    labels[300:] = 2
    return np.round(data, 3), labels
#dataSet为坐标值,labels为类别
dataSet, labels = init_data()


#分类函数,input为输入的点坐标
def classify(input, dataSet, labels, k):
    dataSetLen = dataSet.shape[0]
    #将输入点平铺成与训练集大小相同的数组
    input = np.tile(input, (dataSetLen, 1))
    #广播计算(欧式距离公式)
    distances = ((input - dataSet)**2).sum(axis=1)**0.5
    #依据distances排序，输出其索引值.
    sortedIndex = distances.argsort()
    classNum = {1: 0, 2: 0}
    for i in range(k):
        #第k个点的距离
        pointLabel = labels[sortedIndex[i]]
        classNum[pointLabel] = classNum[pointLabel] + 1
    maxClass = max(classNum.items(), key=lambda x: x[1])[0]
    return maxClass

#迭代获得最佳K的取值
rightNum = 0
sums = 0
accuracyList = []
kList = []
for k in range(1, 350, 5):
    #每个训练十次
    for j in range(10):
        #每次打乱下一下数据集
        shuffleIndex = np.random.permutation(np.arange(len(dataSet)))
        randomData = dataSet[shuffleIndex]
        randomLabel = labels[shuffleIndex]
        training_data = randomData[0:450]
        training_lable = randomLabel[0:450]
        test_data = randomData[450:]
        test_label = randomLabel[450:]

        for i in range(50):
            outLabel = classify(test_data[i], training_data, training_lable, k)
            if outLabel == test_label[i]:
                rightNum = rightNum + 1
        # accuracy.append(rightNum/50)
        sums = sums + rightNum
        rightNum = 0
    accuracyList.append(sums / 500)
    kList.append(k)
    sums = 0
print("最佳K取值为:", kList[np.argmax(accuracyList)], "正确率为:", max(accuracyList))
bestK = kList[np.argmax(accuracyList)]



x_train = dataSet
y_train = labels
#根据最小值划定区域
x_min, y_min = dataSet.min(axis=0)
x_max, y_max = dataSet.max(axis=0)
rangeX = np.linspace(x_min, x_max, 200)
rangeY = np.linspace(y_min, y_max, 200)
# 生成网格采样点
xList, yList = np.meshgrid(rangeX, rangeY)
rangeMap = np.stack((xList.flat, yList.flat), axis=1)
#使用最佳K对待定点进行划分
predictLabel = []
for i in range(len(rangeMap)):
    predictLabel.append(classify(rangeMap[i], dataSet, labels, bestK))

LabelList = np.array(predictLabel).reshape(xList.shape)
LabelList = LabelList.astype(float)

# 绘制生成的a,b点分布图
plt.figure()
x1, y1 = dataSet[0:300].T  
x2, y2 = dataSet[300:500].T  
plt.scatter(x1, y1, c='y', marker='o', alpha=0.5)
plt.scatter(x2, y2, c='r', marker='o', alpha=0.5)
plt.title("A,B两类点的分布图")
plt.xlabel("x")
plt.ylabel("y")

plt.figure()
x = range(len(kList))
y = accuracyList
plt.plot(x, y, marker='o')
plt.xticks(x, kList, rotation=70)
plt.xlabel("k")  
plt.ylabel("正确率")  
plt.title("不同K取值的正确率曲线图")

plt.figure()
color_map = ListedColormap(["g", 'b'])
plt.pcolormesh(xList, yList, LabelList, cmap=color_map)
plt.scatter(x1, y1, c='y', marker='o', alpha=0.5)
plt.scatter(x2, y2, c='r', marker='o', alpha=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("K={}时的分类效果图".format(bestK))
plt.grid(True)
plt.show()