from pylab import *
# from sklearn.svm import SVC
# import warnings
# from sklearn.model_selection import GridSearchCV


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
# 绘制生成的a,b点分布图
plt.figure()
x1, y1 = dataSet[0:300].T  
x2, y2 = dataSet[300:500].T  
plt.scatter(x1, y1, c='y', marker='o', alpha=0.5)
plt.scatter(x2, y2, c='r', marker='o', alpha=0.5)
plt.title("A,B两类点的分布图")
plt.xlabel("x")
plt.ylabel("y")

# 绘制生成的原始ab点分布图
def plotPoints(dataSet):
    plt.scatter(dataSet[0:300].T[0],dataSet[0:300].T[1], c = 'r', marker = '.')
    plt.scatter(dataSet[300:500].T[0],dataSet[300:500].T[1], c = 'b', marker = 'x')
    plt.axis()
    plt.title("A&B points distribution")
    plt.xlabel("x")
    plt.ylabel("y")


# 欧氏距离计算
def distEuclid(x, y):
    return np.sqrt(np.sum((x - y) ** 2))  # 计算欧氏距离

# 随机生成初始中心点
def randomCenters(dataSet, k):
    m, n = dataSet.shape # m = 500, n = 2
    clusterCenters = np.zeros((k, n))#生成2,2的数组

    for i in range(k):
        index = int(np.random.uniform(0, m))  # 0-500间任意选择一个点
        clusterCenters[i, :] = dataSet[index, :] # 让簇中心点等于随机点
    #返回k行2点的数组
    return clusterCenters

# 类心偏移距离作为cost 绘制曲线图
def costPlot(costList):
    plt.figure()
    x = range(100)
    y = costList[0:100]
    plt.plot(x,y, c='b', marker='o',mec='r', mfc='w', alpha=0.5)
    plt.xlabel(u"epoch number")  # X轴标签
    plt.ylabel("cost")  # Y轴标签

# k均值聚类训练
def KMeans(dataSet, k):
    m = np.shape(dataSet)[0] # 样本点个数

    # pointsMat第一列存样本属于哪一簇,第二列放置样本的到簇的中心点的误差，用距离的平方表示
    pointsMat = np.mat(np.ones((m, 2))*(-1)) # 初始化矩阵

    clusterChange = True
    # 第一步. 调用随机方法生成随机中心点
    clusterCenters = randomCenters(dataSet, k)

    zeroCost = []
    oneCost = []
    count = 0
    # 对所有点遍历过程中，有任何一个点改变了聚类，就设置成True，将开启下一轮遍历
    while clusterChange:
        count += 1 # 迭代轮数
        clusterChange = False
        # 遍历所有的样本
        for i in range(m):
            minDist = 1000
            clusterIndex = -1
            # 第二步：遍历所有的类心,找出最近的类心，归类
            for clusterClass in range(k):
                # 计算该样本点到类心的欧式距离
                distance = distEuclid(clusterCenters[clusterClass, :], dataSet[i, :])
                # 如果
                if distance < minDist:
                    minDist = distance
                    clusterIndex = clusterClass
            
            # 第三步：更新样本所属的类,如果发生变化就,对迭代标志True
            if pointsMat[i, 0] != clusterIndex:
                clusterChange = True # 
                pointsMat[i, :] = clusterIndex, minDist**2
            
            # 第四步：每次算完一行都更新一下类心
            # 原类心横坐标
            clusterX = clusterCenters[clusterIndex][0] 
            clusterY = clusterCenters[clusterIndex][1]

            thisClassPoint = dataSet[np.nonzero(pointsMat[:, 0].A == clusterIndex)[0]]  # 获取一个cluster中所有的点

            # 返回更新后的类心
            clusterCenters[clusterIndex, :] = np.mean(thisClassPoint, axis=0) # 对矩阵的行求均值

            # 把每次迭代类心点的偏移距离作为cost
            cost = sqrt((clusterCenters[clusterIndex][0] - clusterX) ** 2 + (clusterCenters[clusterIndex][1] - clusterY) ** 2)
            if (clusterIndex == 0):
                zeroCost.append(cost)
            else:
                oneCost.append(cost)
        # 得到迭代完一轮以后得到的类心
        for j in range(k):
            thisClassPoint = dataSet[np.nonzero(pointsMat[:, 0].A == j)[0]]  
            clusterCenters[j, :] = np.mean(thisClassPoint, axis=0)  

    costPlot(zeroCost)
    plt.title("A类中心点变化趋势图")
    costPlot(oneCost)
    plt.title("B类中心点变化趋势图")
    print("clusterCenter坐标 ：",clusterCenters[0],clusterCenters[1])
    print("迭代次数：%d 轮 ，每轮500次共计 %d 次"%(count,count*500))
    return clusterCenters, pointsMat
# 绘制聚类图以及类心点
def showResult(dataSet, labelSet, k, clusterCenters, pointsMat):
    m, n = dataSet.shape
    if(clusterCenters[0][0]>clusterCenters[1][0]):
        mark = ['or', 'oy']
    else:
        mark = ['oy', 'or']
    plt.figure()
    correctNum = 0 # 分类正确的样本点个数
    # 绘制所有的样本
    for i in range(m):
        markIndex = int(pointsMat[i, 0]) # 第i个点所属的类别序号
        if((clusterCenters[0][0]<clusterCenters[1][0] and markIndex == labelSet[i])
                or(clusterCenters[0][0]>clusterCenters[1][0] and 1-markIndex == labelSet[i])):
            correctNum += 1
        plt.title('聚类结果')
        plt.plot(dataSet[i, 0],dataSet[i, 1], mark[markIndex], alpha=0.5)

    accuracy = correctNum/500
    print('正确率为 : ',accuracy*100,'%')
    mark = ['Dk', 'Dg']
    # 绘制类心点
    for i in range(k):
        plt.plot(clusterCenters[i, 0], clusterCenters[i, 1], mark[i])

k = 2
clusterCenters, pointsMat = KMeans(dataSet,2)
showResult(dataSet, labels, k, clusterCenters, pointsMat)
plt.show()