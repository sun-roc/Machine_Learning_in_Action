import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

def print_table(Matrix):
    movieTable = PrettyTable(['Lady in the Water', 'Snake on a Plane', 'Just My Luck',
                              'Superman Returns', 'You, Me and Dupree', 'The Night Listener'])
    Matrix = np.round(Matrix, 3)
    for i in range(7):
        movieTable.add_row(Matrix[i])
    movieTable.add_column( "name",['Lisa Rose', 'Gene Seymour', 'Michael Phillips',
                                     'Claudia Puig', 'Mick LaSalle', 'Jack Matthews', 'Toby'])
    print(movieTable)

def gradient_descent(scoringMatrix, Matrix_1, Matrix_2, K, epoch=2000, alpha=0.0002, beta=0.02,threshold = 0.001):
    row = len(scoringMatrix)
    column = len(scoringMatrix[0])
    Matrix_2 = Matrix_2.T
    loss = []
    for step in range(epoch):
        error = 0
        for i in range(row):
            for j in range(column):
                error_ij = scoringMatrix[i][j] - np.dot(Matrix_1[i, :], Matrix_2[:, j])
                for k in range(K):
                    if scoringMatrix[i][j] > 0:
                        Matrix_1[i][k] = Matrix_1[i][k] + alpha * \
                            (2*error_ij*Matrix_2[k][j]-beta*Matrix_1[i][k])
                        Matrix_2[k][j] = Matrix_2[k][j] + alpha * \
                            (2*error_ij*Matrix_1[i][k]-beta*Matrix_2[k][j])  
                # 求均方差
                if scoringMatrix[i][j] > 0:
                    error = error + pow(scoringMatrix[i][j] -
                                np.dot(Matrix_1[i, :], Matrix_2[:, j]), 2)  # 损失值的和
                    for k in range(K):
                        error = error + (beta / 2) * \
                            (pow(Matrix_1[i][k], 2) + pow(Matrix_2[k][j], 2))
        loss.append(error)
        if error < threshold:  # 收敛条件
            break

        recommendMatrix = np.dot(Matrix_1, Matrix_2)
    return recommendMatrix,loss


critics = {'Lisa Rose': {'Lady in the Water': 2.5, 'Snake on a Plane': 3.5,
                         'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
                         'The Night Listener': 3.0},

           'Gene Seymour': {'Lady in the Water': 3.0, 'Snake on a Plane': 3.5,
                            'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
                            'You, Me and Dupree': 3.5, },

           'Michael Phillips': {'Lady in the Water': 2.5, 'Snake on a Plane': 3.0,
                                'Superman Returns': 3.5, 'The Night Listener': 4.0},

           'Claudia Puig': {'Snake on a Plane': 3.5, 'Just My Luck': 3.0,
                            'The Night Listener': 4.5, 'Superman Returns': 4.0,
                            'You, Me and Dupree': 2.5},

           'Mick LaSalle': {'Lady in the Water': 3.0, 'Snake on a Plane': 4.0,
                            'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
                            'You, Me and Dupree': 2.0},

           'Jack Matthews': {'Lady in the Water': 3.0, 'Snake on a Plane': 4.0,
                             'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},

           'Toby': {'Snake on a Plane': 4.5, 'You, Me and Dupree': 1.0, 'Superman Returns': 4.0}}

scoringMatrix = [
    [2.5, 3.5, 3, 3.5, 2.5, 3],
    [3, 3.5, 1.5, 5, 3.5, 3],
    [2.5, 3, 0, 3.5, 0, 4],
    [0, 3.5, 3, 4, 2.5, 4.5],
    [3, 4, 2, 3, 2, 3],
    [3, 4, 0, 5, 3.5, 3],
    [0, 4.5, 0, 4, 1, 0]
]

print("用户打分表：")
print_table(scoringMatrix)


scoringMatrix = np.array(scoringMatrix)
row = len(scoringMatrix) 
column = len(scoringMatrix[0])  
K = 3 # K值可变
Matrix_1 = np.random.rand(row, K)  
Matrix_2 = np.random.rand(column, K)  
recommendMatrix,loss = gradient_descent(scoringMatrix, Matrix_1, Matrix_2, K)

print("Matrix_1 \n",Matrix_1)
print("Matrix_2 \n",Matrix_2)
print("推荐打分表：")
print_table(recommendMatrix)
recommendMatrix = np.round(recommendMatrix, 3)
name = ['Lady in the Water', 'Just My Luck', 'The Night Listener']
TobyTable = PrettyTable(name)
TobyTable.add_row([recommendMatrix[6][0], recommendMatrix[6][2], recommendMatrix[6][5]])
print("预测Toby对电影的评分：\n", TobyTable)
plt.plot(range(len(loss)), loss)
plt.ylabel("loss")
plt.show()



