2个目标：
**Basic Natural Language Processing**;
**Deep Learning for Text Understanding**;

干了什么：
分析给定的25000条数据(id,sentiment,review),观察review并预测sentiment是积极的(positive)或是消极的(negative)；

评估：
Submissions are judged on area under the ROC curve

提交：
提交逗号分隔的文件，包含：
行：25000行和1个标题行；
列：id和sentiment；sentiment为二元预测：1为positive，0为negative；

深度学习(DL)：
In this tutorial, we use a hybrid approach to training -- consisting of an unsupervised piece (Word2Vec) followed by supervised learning (the Random Forest).

----------------------------PART：1------------------------------------------
PART1 model:
词袋模型：Bag of Words model;

dataset:
**labeledTrainData**, which contains 25,000 IMDB movie reviews, each with a positive or negative sentiment label.

-----------------------------处理步骤-----------------------------------------
读取dataset：
```python
import pandas as pd
train=pd.read_csv("labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
# 不设置quoting参数，默认会去除英文双引号，只留下引号内的内容；设置quoting=3，会如实读取数据，即数据包括引号
# delimiter="\t":每一条数据的各个column间以制表符分割
print(train.shape)
# (25000,3)
print(train.columns.values)
# [id,sentiment,review]
print(train['review'][0])
# review列的第一条记录，过长，自己看；会有类似"<br/>"之类的标签所以需要清洗
```

清洗数据和文本预处理:
```python
# 借助BeautifulSoup4库进行文本清洗,sudo pip install BeautifulSoup4
from bs4 import BeautifulSoup4
example = BeautifulSoup(train['review'][0])
print(example.get_text())#调用函数清除标签如"<br/>"
# 可以用print(train['review'][0])做对比进行查看
```

对于标点符号，数字或停用词：
考虑由于一些符号如"!!"带有感情色彩，所以应当视作words来考虑而不是消去，但本例为简单选择移除符号，数字；

补充：安装nltk及nltk_data
```python
conda activate 你的环境
pip install nltk

进入python环境后执行：
nltk.download()

上述语句执行后选择的存放位置应该包含在:
nltk.data.path中，具体可以通过：
print(nltk.data.path)查看，本机为：
['/Users/lihao/nltk_data', '/opt/anaconda3/envs/NLP/nltk_data', '/opt/anaconda3/envs/NLP/share/nltk_data', '/opt/anaconda3/envs/NLP/lib/nltk_data', '/usr/share/nltk_data', '/usr/local/share/nltk_data', '/usr/lib/nltk_data', '/usr/local/lib/nltk_data']

安装成功后测试：
from nltk.book import *
查看输出结果；
```

```python
import re
# 利用正则库re来将非字符替换为空格
letters_only = re.sub("[^a-zA-Z]"," ",example.get_text())
print(letters_only)
# 将字符转换为小写，并分割为单词
lower_case = letters_only.lower()
words = lower_case.split()

# 处理如:"a","and","is","the"之类的停用词
# import nltk
# nltk.download()# Download text data sets, including stop words
from nltk.corpus import stopwords
# print(stopwords.words("english"))# 查看英语的停用词
words = [w for w in words if not w in stopwords.words("english")]
# 取出非停用词
print(words)
```

定义一个函数实现清洗功能：
```python
def review_to_words(raw_review):
	review_text = BeautifulSoup(raw_review).get_text()
	letters_only = re.sub("[^a-zA-Z]"," ",review_text)
	words = letters_only.lower().split()
	# python中搜索set比list快
	stops = set(stopwords.words("english"))
	meaningful_words = [w for w in words if not w in stops]
	# 将筛完的词以空格拼接形成string,即列表元素每两个之间加一个空格
	return (" ".join(meaningful_words))
```

清理所有的train-set：
```python
num_reviews = train['review'].size
print("清洗并解析训练集中的电影评论")
clean_train_reviews = []
for i in range(0,num_reviews):
	if((i+1)%1000==0):
		print("review %d of %d\n"%(i+1,num_reviews))
	clean_train_reviews.append(review_to_words(train['review'][i]))
```

Bag of Words(词袋模型)：
将文本表示为词的集合，忽略词语的顺序和语法结构，只关注词频。即将文本表示为词频向量；
词频矩阵：$a[i][j]$表示第j个词在第i个文本中出现的次数；
目的：将非结构化的文本数据转换为结构化的数值数据,便于机器学习模型训练和预测

使用feature_extraction 模块(from scikit-learn)来创建bag-of-words features;
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)
# 对于每个训练文本，只考虑每种词汇在该训练文本中出现的频率，将文本中的词语转换为词频矩阵，通过fit_transform计算各个词语出现的次数

train_data_features = vectorizer.fit_transform(clean_train_reviews)
# 将训练集转换为feature vectors，输入是string列表
train_data_features = train_data_features.toarray()
# print(train_data_features.shape)
# (25000,5000):5000 features,one for each vocabulary word

'''
CountVectorizer自带预处理(preprocessor)，标记化(tokenizer)，停用词删除(stop_words),可以使用自带的参数或自己实现相应功能；

fit_transform:将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示第j个词在第i个文本下的词频

get_feature_names:可看到所有文本的关键字，通过toarray()看到词频矩阵结果
get_feature_names_out(字符串列表):同上，使用toarray()查看词频矩阵结果
'''
```
查看vocabulary：
```python
# vocab = vectorizer.get_feature_names()
# print(vocab)
上述方式已过时，更新方式如下：
vocab = vectorizer.get_feature_names_out(字符串列表)
print(vocab)

打印每个词及其词频：--一个小例子
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

l =["hello cat","hello dog","cat is good","dog is vicious","I like cat","I hate dog"]
vectorizer = CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)

train_data_feature = vectorizer.fit_transform(l)
# fit_transform:先拟合数据，后将其转换为标准形式，用于训练集
# transform:作用于测试集，因为已经找到数据的均值方差，所以只是直接标准化
vocab = vectorizer.get_feature_names_out(l)
dist = np.sum(train_data_feature.toarray(),axis=0)# axis=0按列求和
for name,num in zip(vocab,dist):
	print(name,num)
'''
cat 3 
dog 3 
good 1 
hate 1 
hello 2 
is 2 
like 1 
vicious 1
'''
查看对应的词频矩阵：--矩阵元素含义见上注释
print(train_data_feature.toarray())
'''
word:cat dog good hate hello is like vicious
l:	[
l0	[1 0 0 0 1 0 0 0] 
l1	[0 1 0 0 1 0 0 0] 
l2	[1 0 1 0 0 1 0 0] 
l3	[0 1 0 0 0 1 0 1] 
l4	[1 0 0 0 0 0 1 0] 
l5	[0 1 0 1 0 0 0 0]]
'''
```

随机森林：
```python
from sklearn.ensemble import RandomForestClassifier
# ensemble为集成算法，如随机森林
# 初始化100棵随机数
forest = RandomForestClassifier(n_estimators=100)
# 增大随机森林的个数可能会提高性能，但会极大增加训练时间
forest = forest.fit(train_data_features,train["sentiment"])
```

对测试集进行预测：
```python
test = pd.read_csv("testData.tsv",header=0,delimiter="\t",quoting=3)
# print(test.shape)
# (25000,2)
clean_test_reviews = []
print("清洗并解析测试集中的电影评论")
for i in range(0,num_reviews):
	if((i+1)%1000==0):
		print("review %d of %d\n"%(i+1,num_reviews))
	clean_review = review_to_words(test["review"][i])
	clean_test_reviews.append(clean_review)

# 注意此处调用transform，注意区分；
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)

# 输出到文件
output = pd.DataFrame(data={
							"id":test["id"],
							"sentiment":result})
output.to_csv("Bag_of_Words_model.csv",index=False,quoting=3)
```

完整代码：
```python
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

def review_to_words(raw_review):
	review_text = BeautifulSoup(raw_review).get_text()
	letters_only = re.sub("[^a-zA-Z]"," ",review_text)
	words = letters_only.lower().split()
	stops = set(stopwords.words("english"))
	meaningful_words = [w for w in words if not w in stops]
	return (" ".join(meaningful_words))

# train = pd.read_csv("labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)

# test = pd.read_csv("testData.tsv",header=0,delimiter="\t",quoting=3)

# 注意加入kaggle中dataset的位置，否则无法读取输入数据，如/kaggle/.../..
train = pd.read_csv("train.csv",header=0)
test = pd.read_csv("test.csv",header=0)
  

num_reviews = train["review"].size
print("清洗并解析训练集中的电影评论")
clean_train_reviews = []

for i in range(0,num_reviews):
	if((i+1)%1000==0):
		print("review %d of %d\n"%(i+1,num_reviews))
	clean_review = review_to_words(train["review"][i])
	clean_train_reviews.append(clean_review)

  
vectorizer = CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)

train_data_features = train_data_features.toarray()

forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features,train["label"])

clean_test_reviews = []
num_reviews = test["review"].size
print("清洗并解析测试集中的电影评论")
	for i in range(0,num_reviews):
	if((i+1)%1000==0):
		print("review %d of %d\n"%(i+1,num_reviews))
	clean_review = review_to_words(test["review"][i])
	clean_test_reviews.append(clean_review)

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
  
result = forest.predict(test_data_features)

output = pd.DataFrame(data={"Id":[num for num in range(0,result.size)],"Predicted":result})
output.to_csv("Bag_of_Words_model.csv",index=False,quoting=3)
```
结果：
![[Pasted image 20240917094835.png]]
将随机森林的初始个数从100改为500，性能略微提升，但训练时间急剧增大；


----------------------------PART：2------------------------------------------
PART2 model:
词向量模型：word vectors;

data-set：
**unlabeledTrain.tsv**, which contains 50,000 additional reviews with no labels.since Word2Vec can learn from unlabeled data, these extra 50,000 reviews can now be used.
-----------------------------处理步骤-----------------------------------------
读取数据并查看维度：
```python
import pandas as pd
train = pd.read_csv("labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
print(train.shape)
print(unlabeled_train.shape)
print(test.shape)
# (25000, 3)
# (50000,2)--没有标签列
# (25000, 2)
```

清理数据:（注意，由于word2vec算法，此时不要清理停用词）
```python
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def review_to_wordlist(review,remove_stopwords=False):
	# 清理html
	review_text = BeautifulSoup(review).get_text()
	# 去除所有非字母字符
	review_text = re.sub("[^a-zA-Z]"," ",review_text)
	# 将所有字母转换为小写并分割
	words = review_text.lower().split()
	# 默认不移除停用词
	if remove_stopwords==True:
		stops = set(stopwords.words("english"))
		words = [w for w in words if not w in stops]
	# 返回words的列表
	return words
```

解决word2vector输入问题：需要单个句子，每个句子是单词的列表，所以输入应该是列表的列表；使用nltk的punkt tokenizer实现句子划分；
```python
import nltk.data
# 加载punkt tokenizer
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

def review_to_sentences(review,tokenizer,remove_stopwords=False):
# 返回一个list，分词；strip()去除首尾空格
raw_sentences = tokenizer.tokenize(review.strip())
sentences = []
for raw_sentence in raw_sentences:
	if len(raw_sentence)>0:
		sentences.append(
		review_to_wordlist(raw_sentence,remove_stopwords))
return sentences
```

利用定义的函数为word2vector准备输入：
```python
sentences = []
print("从训练集中解析句子")
for review in train["review"]:
	sentences += review_to_sentences(review,tokenizer)

print("从未标记的训练集中解析句子")
for review in unlabeled_train["review"]:
	sentences += review_to_sentences(review,tokenizer)
```
此处对列表的”+=“和append不可以互换：测试样例如下：
```python
sub_l = [[1,2,3],[4,5,6]]
l = [[7,8,9],[10,11,12]]
l+=sub_l
print(l)
'''
[[7, 8, 9], [10, 11, 12], [1, 2, 3], [4, 5, 6]]
'''
sub_l = [[1,2,3],[4,5,6]]
l = [[7,8,9],[10,11,12]]
l.append(sub_l)
print(l)
'''
[[7, 8, 9], [10, 11, 12], [[1, 2, 3], [4, 5, 6]]]
'''
即+=是将list_of_list中的每一个元素放到另一个list_of_list中，而append是将整个list_of_list作为一个元素放到另一个list_of_list中；
```
使用gensim中的word2vec模型，初始化参数：特征数量，最小词数，下采样参数，上下文窗口大小，工作线程数；调用word2vec.Word2Vec,定义模型的名字并使用save方法保存模型;
```python
num_features=300
min_words_count =40
downsampling=1e-3
context= 10
num_workers = 4

from gensim.models import word2vec
print("training model...")
model = word2vec.Word2Vec(sentences,workers=num_workers,vector_size=num_features,sample =downsampling,window=context,min_count = min_words_count)

model.init_sims(replace=True)
model_name = "300features_40minwords_10context"
model.save(model_name)
# model.load(model_name)---加载模型
'''
一个值得注意的点是Word2Vec由于版本更新：
size参数变为vector_size；
iter参数变为epochs
'''
```

补充：
使用top指令可以在terminal中观察到并行运行的情况，如：
```
top -o cpu
```
![[Pasted image 20240918085453.png]]
如上图，第一条应该为python，且数值在300～400之间；

使用model.wv.doesnt_match(str)方法判断str中最不同的字符串：
```python
print(model.wv.doesnt_match("man woman child kitchen".split()))
# kitchen
```
模型训练完后使用wv属性访问(word vector)，如下：
```python
word_vector = model.wv
print(word_vector)
# KeyedVectors<vector_size=300, 16490 keys>
# 即由于设置min_word_count,所以共有16490个词，每个词特征有300个
# word_vector.vectors.shape可以查看维度
'''
word_vector.save(name)--保存词向量

from gensim.models import KeyedVectors
wv = KeyedVectors.load(name,mmap="r")--加载词向量
'''
几个方法：
print(word_vector.vectors)--打印词向量矩阵
print(word_vector.index_to_key)--查看所有词汇
print(word_vector.key_to_index)--查看词汇的索引
for word in word_vector.index_to_key:
	print(word,word_vector.get_vecattr(word,'count'))
	--查看词出现的次数

查看相近的词：
word_vector.similar_by_word(word)
word_vector.similar_by_key(key)

查看相似度：
word_vector.similarity(word1,word2)

推断相似词：
word_vector.most_similar(positive=[word1,word2],negative=[word])

最不相似词：
word_vector.doesnt_match(word_list)
```

----------------------------PART：3------------------------------------------
分为两个部分，PART1利用词向量的均值和随机森林来预测，输入数据为打上标签的labeledtraindata；PART2利用clustering聚类来预测，输入数据为unlabeltraindata；
-----------------------------处理步骤-----------------------------------------
加载PART2的模型,并查看词矩阵或词向量的维度，查看单个词的词向量信息：
```python
from gensim.models import Word2Vec

model = Word2Vec.load("300features_40minwords_10context")
print(model.wv["flower"])---打印flower的词向量
# print(model.wv['flower'].shape)---(1,300)
print(model.wv.vectors.shape)
# (16490, 300),行：词数 列：特征数
'''
注意，从gensim的4.0版本开始，word2vec模型不再支持如model["str"]下标索引，需要通过wv属性进行操作；
'''
```

-----------from words to paragraphs,Attempt 1:vector averaging:---------------

给定一个paragraph，对其中的words的词向量求平均：
```python
import numpy as np

def makeFeatureVec(words,model,num_features):
	featureVec = np.zeros((num_features,),dtype="float32")
	nwords = 0  # 一个句子中的单词个数
	index2word_set = set(model.wv.index_to_key)
	# 包含模型词汇库中的词
	for word in words:# 对句子中出现在词汇库中的词特征向量求平均
		if word in index2word_set:
			nwords += 1
			featureVec = np.add(featureVec,model.wv[word])
	featureVec = np.divide(featureVec,nwords)
	return featureVec
'''
由于gensim高版本，所以部分函数有变动，如model.wv.index_to_key
'''
```

对所有reviews求词向量均值操作：
```python
def getAvgFeatureVecs(reviews,model,num_features):
	# 二维list，最内层是单个review的word列表，外层是reviews集合
	counter = 0
	reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
	for review in reviews:
		if counter%1000==0:
			print ("review %d of %d" %(counter,len(reviews)))
			reviewFeatureVecs[counter] = makeFeatureVec(review,model,num_features)
			counter+=1
	return reviewFeatureVecs
```

分别对训练集和测试集清洗数据，并求reveiws的词向量平均
```python
clean_train_reviews = []

for review in train["review"]:
	clean_train_reviews.append(\				   review_to_wordlist(review,remove_stopwords=True))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews,model,num_features)

print("创建测试集评论的平均特征向量")
clean_test_reviews = []

for review in test["review"]:
	clean_test_reviews.append(\					  review_to_wordlist(review,remove_stopwords=True))

testDataVecs = getAvgFeatureVecs(clean_test_reviews,model,num_features)
```

用随机森林进行训练，输入为reviews的词向量平均值和标签sentiment，并预测结果：
```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100)
print("将labeled training data拟合到随机森林...")

forest = forest.fit(trainDataVecs,train["sentiment"])
result = forest.predict(testDataVecs)

output = pd.DataFrame( data={"Id":test["id"], "sentiment":result} )
output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )
```
![[Pasted image 20240918110003.png]]
总结：性能比Bag of Words低几个百分点；

-----------from words to paragraphs,Attempt 2:clustering----------------------

Trial and error suggested that small clusters, with an average of only 5 words or so per cluster, gave better results than large clusters with many words.
即将K值设置为每5个单词一个簇，或者设置为整个词汇表的1/5；

用词向量初始化K-means聚类，并计时：
```python
from sklearn.cluster import KMeans
import time

start = time.time()

word_vector = model.wv.vectors
# 词向量
num_clusters = int(word_vector.shape[0]/5)
# 聚类的个数，据要求设置为词汇表大小的1/5
# 大小为3298

Kmeans_clustering = KMeans(n_clusters=int(num_clusters))
# 创建聚类并设置聚类的个数

idx = Kmeans_clustering.fit_predict(word_vector)
# fit_predict()接受一个特征矩阵，训练K-means模型并返回每个样本的聚类标签
# 区分fit():无返回值，但会更新K-means对象的内部状态，使其包含训练后的参数
# idx现在存储每个word的聚类标签
# idx.size == 16490,即词汇表的单词数

end = time.time()

elapsed = end-start
# 用于计算运行时间
print("time taken for K-means clustering:",elapsed,"seconds")
```

将word2vec模型中的词汇和聚类标签组合为字典
```python
word_centroid_map = dict(zip(model.wv.index_to_key,idx))
# k:词汇表中的词，长度为:16490
# v:idx中存储的聚类标签,大小为16490,即等于词汇表的大小
```

打印前10个cluster看看，注意用k,v结合for-loop访问字典,将同一cluster的词加入到列表里并打印查看；
```python
for cluster in range(0,10):
	print("\nCluster %d"%cluster)
	words = []
	for k,v in word_centroid_map.items():
		if v==cluster:
			words.append(k)
	print(words)
```

现在为词汇表中的每个词都创建了一个聚类，接下来要将reviews转换到不同的聚类中；
```python
def create_bag_of_centroids(wordlist,word_centroid_map):
	num_centroids = max(word_centroid_map.values())+1
	# 聚类中心的个数
	# 字典的v值(即生成词表聚类的个数)，+1个
	bag_of_centroids = np.zeros(num_centroids,dtype="float32")
	for word in wordlist:
		if word in word_centroid_map:
		# 如果词在某个聚类中
			index = word_centroid_map[word]
			# 记录下这个词对应的聚类编号
			bag_of_centroids[index]+=1	
			# 将这个聚类编号对应的个数加1--即该聚类的元素个数+1
	return bag_of_centroids
```

对训练集和测试集进行清理(同之前，但是移除停用词)，然后创建train中心和test中心；
```python
clean_train_reviews = []
for review in train["review"]:
	clean_train_reviews.append(\					   review_to_wordlist(review,remove_stopwords=True))

clean_test_reviews = []
for review in test["review"]:
	clean_test_reviews.append(\		  review_to_wordlist(review,remove_stopwords=True))

train_centroids = np.zeros((train['review'].size,int(num_clusters)),dtype="float32")
# 二维数组，外层每条review对应一个list；内层大小为聚类的个数
counter = 0
for review in clean_train_reviews:
	train_centroids[counter] = create_bag_of_centroids(review,word_centroid_map)
	counter+=1

test_centroids = np.zeros((test['review'].size,int(num_clusters)),dtype="float32")
counter = 0
for review in clean_test_reviews:
	test_centroids[counter] = create_bag_of_centroids(review,word_centroid_map)
	counter+=1
```

利用随机森林进行预测：
```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100)
print("Fitting a random forest to labeled training data...")
forest = forest.fit(train_centroids,train["sentiment"])
result = forest.predict(test_centroids)

output = pd.DataFrame(data={"Id":test["id"], "sentiment":result})
output.to_csv( "BagOfCentroids.csv", index=False, quoting=3 )
```
![[Pasted image 20240918164540.png]]
总结：与Bag of Words相比，性能相当(或稍弱).

