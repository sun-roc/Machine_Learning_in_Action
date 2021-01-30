import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
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
trainingSet = np.vstack((dataSet[0:240],dataSet[300:460]))
testSet = np.vstack((dataSet[240:300], dataSet[460:500]))
trainingLabels =list(labels[0:240] ) + list(labels[300:460])
testLabels = list(labels[240:300]) + list(labels[460:500])

x1, y1 = dataSet[0:300].T # 所有A类点
x2, y2 = dataSet[300:500].T # 所有B类点
# 绘制生成的a,b点分布图
plt.figure()
x1, y1 = dataSet[0:300].T  
x2, y2 = dataSet[300:500].T  
plt.scatter(x1, y1, c='y', marker='o', alpha=0.5)
plt.scatter(x2, y2, c='r', marker='o', alpha=0.5)
plt.title("A,B两类点的分布图")
plt.xlabel("x")
plt.ylabel("y")


# 线性核
# C：错误项的惩罚系数，C越大泛化能力越弱，越容易过拟合，C跟松弛向量有关
parameters = {'C': np.linspace(0.1,10,50)} 
#寻找最佳惩罚系数取值
clf1 = GridSearchCV(SVC(kernel='linear'), parameters, scoring='f1') # 选择最佳参数
clf1.fit(trainingSet, trainingLabels)  # 训练
print('线性核的最佳参数为 : ',clf1.best_params_)
clf1 = SVC(kernel='linear', C=clf1.best_params_['C'])
clf1.fit(trainingSet, trainingLabels)
print("最佳线性核参数在测试集上的正确率为 : ",clf1.score(testSet,testLabels)*100,"%")



# 绘制用linear核的SVM得到的超平面图
def plot_linear_hyperplane(clf, title='hyperplane'):
    plt.figure()
    x1, y1 = dataSet[0:300].T  
    x2, y2 = dataSet[300:500].T  
    plt.scatter(x1, y1, c='y', marker='o', alpha=0.5)
    plt.scatter(x2, y2, c='r', marker='o', alpha=0.5)
    plt.title("线性核的支持向量分类图")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.scatter(clf1.support_vectors_[:, 0],
                   clf1.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')

    # 绘制决策函数
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # 创建网格来评估模型
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    XX, YY = np.meshgrid(xx, yy)
    
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    
    # 绘制决策边界和边距
    ax.contour(XX, YY, Z, colors = 'k', levels = [-1, 0, 1], alpha = 0.5, linestyles=['--', '-', '--'])
    # 绘制支持向量（Support Vectors）
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s = 30)
# 绘制用linear核的SVM得到的超平面图
def plot_hyperplane(clf, title='hyperplane'):
    plt.figure()
    x1, y1 = dataSet[0:300].T  
    x2, y2 = dataSet[300:500].T  
    plt.scatter(x1, y1, c='y', marker='o', alpha=0.5)
    plt.scatter(x2, y2, c='r', marker='o', alpha=0.5)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.scatter(clf1.support_vectors_[:, 0],
                   clf1.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')

    # 绘制决策函数
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # 创建网格来评估模型
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    XX, YY = np.meshgrid(xx, yy)
    
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
   
    Z = clf.decision_function(xy).reshape(XX.shape)   
    # 绘制决策边界和边距
    ax.contour(XX, YY, Z,levels=[-1, 0, 1],cmap=plt.cm.winter, alpha=0.5,linestyles=['--', '-', '--'])

    # 绘制支持向量（Support Vectors）
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s = 30)
 

plot_linear_hyperplane(clf1, title='linear kernel hyperplane')


def nonlinearityKernel(name):
    if(name == "poly"):
        parameters = {'C': np.linspace(0.1,10,30), 'gamma': np.linspace(0.0001,0.5,10),"degree":[1,2]}
    else:
        parameters = {'C': np.linspace(0.1,10,30), 'gamma': np.linspace(0.001,1,10)}
    clf2 = GridSearchCV(SVC(kernel='{}'.format(name)), parameters, scoring='f1') # 选择最佳参数
    clf2.fit(trainingSet, trainingLabels)
    print('{} kernel 的最佳参数为 : '.format(name),clf2.best_params_)
    clf2 = SVC(kernel='{}'.format(name),C=clf2.best_params_['C'],gamma=clf2.best_params_['gamma'])
    clf2.fit(trainingSet, trainingLabels)
    print("最佳{}参数在测试集上的正确率为 : ".format(name),clf2.score(testSet, testLabels)*100,"%")
    plot_hyperplane(clf2, title='{} kernel hyperplane'.format(name))

nonlinearityKernel("rbf")
nonlinearityKernel("sigmoid")
nonlinearityKernel("poly")
plt.show()