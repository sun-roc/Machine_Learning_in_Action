# FederalistPapersClassification-using-NaiveBayesClassifier

Identify the authors of the 11 disputed Federalist Papers using naive Bayes classifier

根据已有的85篇文集,基于朴素贝叶斯方法对这11篇作者成谜的文章进行分类，分成Hamilton和Madison两大类。

# 问题背景

联邦党人文集汇集了亚历山大·汉密尔顿、詹姆斯·麦迪逊和约翰·杰伊在1780年下半年用化名“Publius”撰写的85篇文章和文章，以促进美国宪法的批准。汉密尔顿选择了“Publius”作为写作系列的化名。在汉密尔顿于1804年去世后，他写的一份名单被公之于众，这份名单将大多数文章都归于他本人，包括一些似乎更有可能是麦迪逊的作品。

利用朴素贝叶斯方法对这11篇作者成谜的文章进行分类，分成Hamilton和Madison两大类，来推测它们的作者。

# 文件结构

\1. 《联邦党人文集》的85篇文章。来源为:https://avalon.law.yale.edu/subject_menus/fed.asp从该网站爬取85篇文章并按顺序编号存储在 The-Federalist-Papers.txt 文本文件中.

\2. 主要实验程序为:Bayes.py文件负责预处理数据和进行贝叶斯分类.

# 算法与原理

## 朴素贝叶斯理论

朴素贝叶斯（naive bayes）法是基于贝叶斯定理与特征条件独立假设的分类方法。

优点：在数据较少的情况下仍然有效，可以处理多分类问题。

缺点：对入输入数据的准备方式较为敏感。

朴素贝叶斯分类器是一个概率分类器。现有两类文章Hamilton和Madison,设两类文章为![img](https://gitee.com/sun-roc/picture/raw/master/img/clip_image002.png) 统计两类文章中出现频率较高的词, 然后根据各特征词出现的频率组成一个长的词向量. 然后，在11篇未知作者的文章中依个检索所有特征词，依据这些词分别求得![img](https://gitee.com/sun-roc/picture/raw/master/img/clip_image004.png)和![img](https://gitee.com/sun-roc/picture/raw/master/img/clip_image006.png) 代入贝叶斯公式中依据概率大小进行分类.

贝叶斯公式:![img](https://gitee.com/sun-roc/picture/raw/master/img/clip_image008.png)

其中![img](https://gitee.com/sun-roc/picture/raw/master/img/clip_image010.png)为各特征词出现的频率.根据朴素贝叶斯理论, 把每个特征词的出现看成是独立的。假设每个特征相互独立，不相关。

则：![img](https://gitee.com/sun-roc/picture/raw/master/img/clip_image012.png)

将该公式带入本实验中,可以得到:

![img](https://gitee.com/sun-roc/picture/raw/master/img/clip_image014.png)

![img](https://gitee.com/sun-roc/picture/raw/master/img/clip_image016.png)

原式中分母中的P(X)两个式子都含此项,对比较大小无影响故而省略,仅需比较两个式子的大小,取概率大的作为最终取值.且在计算过程中为了使计算步骤简略,对式子取对数后运算,但对最终结果无影响.

## 词袋模型（Bag Of Words）

前面提到的根据单词出现频率构造特征的方法被称为词袋模型,词袋模型使表示文本特征的一种方式。给定一篇文档，它会有很多特征，比如文档中每个单词出现的次数、某些单词出现的位置、单词的长度、单词出现的频率……而词袋模型只考虑一篇文档中单词出现的频率(次数)，用每个单词出现的频率作为文档的特征。

 

# 具体过程

## 1.  数据预处理

![img](https://gitee.com/sun-roc/picture/raw/master/img/clip_image018.jpg)

如上图所示,首先将文章的txt文本导入,然后根据每篇文章开头都有的No.对文章进行切分,每篇文章作为一个item放到数据列表当中.之后遍历存放文章的列表,按种类将Unknown,Hamilton,Madison三类文章存放到对应种类的列表当中.

## 2.  根据词频挑选特征词并构造特征向量

![img](https://gitee.com/sun-roc/picture/raw/master/img/clip_image020.jpg)

如上图所示,将三个列表拼接后进行特征提取,鉴于运行速度和准确性的考量,出现次数大于10词的单词作为特征词,并将传统文本分类中的如a,the等无实际意义的单词剔除,形成特征词向量,部分特征词表如下图所示:

![img](https://gitee.com/sun-roc/picture/raw/master/img/clip_image022.jpg)

形成的特征词向量维度为: 文章数*特征词数 ,每行内容为特征词在该文章中出现的次数.然后将特征词按输入顺序还原为原本的类别列表.

 

## 3.  对进行特征提取后的向量进行数学计算

![img](https://gitee.com/sun-roc/picture/raw/master/img/clip_image024.jpg)

如上图所示

feature_Hami_prob 统计了Hamilton文章中各特征词出现的频率

feature_Madi_prob 统计了Madison文章中各特征词出现的频率

Hami_prob 计算了Hamilton的文章在总文章中出现的频率

Madi_prob 计算了Madison的文章在总文章中出现的频率

## 4.  利用朴素贝叶斯分类

![img](https://gitee.com/sun-roc/picture/raw/master/img/clip_image026.jpg)

如上图所示,将need_to_classify为待分类的文本的特征词向量,将其按照numpy的计算机制与上一步骤提取的概率相乘可以得到朴素贝叶斯公式中所用的![img](https://gitee.com/sun-roc/picture/raw/master/img/clip_image004.png)和![img](https://gitee.com/sun-roc/picture/raw/master/img/clip_image028.png)

为了方便计算对数值取log后于文本出现的频率![img](https://gitee.com/sun-roc/picture/raw/master/img/clip_image030.png)和![img](https://gitee.com/sun-roc/picture/raw/master/img/clip_image032.png)取log后相加可以得到预测的概率,然后按照概率的大小比较输出最后的预测值.

 

# 结果

![img](https://gitee.com/sun-roc/picture/raw/master/img/clip_image034.jpg)

最终预测的11篇(老师所提供的网站中只有十一篇未知作者的文章)结果如上图所示,预测结果11篇文章均为Madison所著. 查阅资料发现，1964年，Mosteller和Wallance发表了他们的研究成果。他们的结论是，这12篇文章的作者很可能都是麦迪逊。因此本程序的预测结果还是比较准确的.

![img](https://gitee.com/sun-roc/picture/raw/master/img/clip_image036.jpg)

该图展示了,预测是Hamilton的概率比上预测是Madison的概率的比值.