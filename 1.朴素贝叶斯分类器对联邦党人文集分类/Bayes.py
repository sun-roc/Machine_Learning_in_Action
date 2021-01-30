import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#读取联邦党人文集文件
papers = open('The-Federalist-Papers.txt').read()
#将paper按篇划分为列表
split_paper_list = papers.split('FEDERALIST No. ')
#由于文章编号从1开始因此前面插入一位
len(split_paper_list)
#存放各个类别paper的列表
papers_Hami = []
papers_Madi = []
papers_Unknown = []
# 把未知文章的篇数号加到列表中
papers_Unknown_index = []
#遍历85篇paper查找能用的上的三种分类
for i in range(85):
    paper = split_paper_list[i]
    if "HAMILTON OR MADISON" in paper:
        papers_Unknown.append(paper)
        papers_Unknown_index.append(i)
    elif "HAMILTON AND MADISON" in paper:
        pass
    elif "HAMILTON" in paper:
        papers_Hami.append(paper)
    elif "MADISON" in paper:
        papers_Madi.append(paper)

#统计每篇文章的出现次数超过10词的高频词,组成一个词向量
Vectorizer = CountVectorizer(min_df=10)
#把他们合成一个列表
papers_list = papers_Hami + papers_Madi + papers_Unknown
#将列表送入Vectorizer统计高频词和次数
feature = Vectorizer.fit_transform(papers_list).toarray()
#打印出统计出的高频词汇
feature_names = Vectorizer.get_feature_names()
# print("高频词: ",feature_names)
#将分好的feature向量切分为三个种类的列表
feature_Hami = feature[:len(papers_Hami), :]
feature_Madi = feature[len(papers_Hami):len(papers_Hami) + len(papers_Madi), :]
feature_Unknow = feature[len(papers_Hami) + len(papers_Madi):, :]

feature_Hami_sum = feature_Hami.sum(axis=0)
feature_Hami_all = feature_Hami_sum.sum()
feature_Hami_sum = np.array(feature_Hami_sum)
feature_Hami_prob = feature_Hami_sum / feature_Hami_all
# print("feature_Hami_prob :",feature_Hami_prob)
feature_Madi_sum = feature_Madi.sum(axis=0)
feature_Madi_all = feature_Madi_sum.sum()
feature_Madi_sum = np.array(feature_Madi_sum)
feature_Madi_prob = feature_Madi_sum / feature_Madi_all
# print("feature_Madi_prob :",feature_Madi_prob)
Hami_prob = len(papers_Hami) / len(papers_list)
Madi_prob = len(papers_Madi) / len(papers_list)


def classifyBayes(need_to_classify, feature_Hami_prob, feature_Madi_prob,
               Hami_prob):
    #将特征词向量转为01序列
    need_to_classify = np.where(need_to_classify == 0, need_to_classify, 1)
    multi_Hami = need_to_classify * feature_Hami_prob
    multi_Madi = need_to_classify * feature_Madi_prob
    multi_Hami = np.where(multi_Hami != 0, multi_Hami, 1)
    multi_Madi = np.where(multi_Madi != 0, multi_Madi, 1)
    #这里使用了Log函数，方便计算，因为最后是比较大小，所有对结果没有影响。
    log_feature_Hami = np.log(multi_Hami)
    log_feature_Madi = np.log(multi_Madi)
    predict_Haim = sum(log_feature_Hami) + np.log(Hami_prob)
    predict_Madi = sum(log_feature_Madi) + np.log(1 - Hami_prob)
  
    #比较概率大小进行判断
    if predict_Haim > predict_Madi:
        print("Hamilton")
    else:
        print("Madison")
    # print("Hami/Madi = " , predict_Haim/predict_Madi)

#循环输出最后的预测结果
for index in range(11):
    print("No {} 预测结果为: ".format(papers_Unknown_index[index]), end='')
    predict = classifyBayes(feature_Unknow[index], feature_Hami_prob,
                         feature_Madi_prob, Hami_prob)
