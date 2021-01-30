from math import sqrt
from prettytable import PrettyTable

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

# 计算用户之间的相近度
def similarity(tagert_user,compare_user):
    sum = 0
    for item in critics[tagert_user]:
        if item in critics[compare_user]:
            sum += pow(critics[tagert_user][item] - critics[compare_user][item], 2)
    return 1/(1+sqrt(sum))
# 对字典进行归一化计算
def normalized(user):
    recomand = {} 
    movieNum_dic = {} 
    for other_user in critics:
        movieNum = 0 
        scoreSum = 0 
        if other_user != user:
            for movie in critics[other_user]:
                movieNum += 1
                scoreSum += critics[other_user][movie]
            averageScore = scoreSum/movieNum 
            for movie in critics[other_user]:
                critics[other_user][movie] = critics[other_user][movie]/averageScore
                if movie not in critics[user] and movie not in recomand:
                    recomand[movie] = 0
                    movieNum_dic[movie] = 0
    return recomand,movieNum_dic


user = 'Toby'
recomand,movieNum_dic = normalized(user)
for other_user in critics:
    if other_user != user:
        for movie in critics[other_user]:
            if movie in recomand:
                recomand[movie] += similarity(user,other_user) * critics[other_user][movie]
                movieNum_dic[movie] += 1
for movie in recomand:
    recomand[movie] /= movieNum_dic[movie]
print("标准化后的取值: \n",critics)
name = ['Lady in the Water', 'Just My Luck', 'The Night Listener']
TobyTable = PrettyTable(name)
TobyTable.add_row(recomand.values())
print("Toby未观看电影的推荐值：\n", TobyTable)
print("向Toby推荐电影顺序：",sorted(recomand.keys(), reverse=True))



