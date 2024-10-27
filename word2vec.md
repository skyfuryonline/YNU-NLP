[CBOW详解](https://www.bilibili.com/list/watchlater?oid=960944394&bvid=BV1XH4y1D7cJ&spm_id_from=333.1007.top_right_bar_window_view_later.content.click)
[skip-gram详解](https://www.bilibili.com/video/BV1Su411c77X/?spm_id_from=333.999.0.0&vd_source=3ac79ed435a4827c66109984966d124a)
[word2vec详解](https://blog.csdn.net/m0_62965652/article/details/136764709)

one-hot编码（不便于余弦相似度计算）
word_embedding:词向量/词嵌入(将one-hot编码的稀疏矩阵->低维连续向量)

word2vec：
重要假设：文本中离得越近的词语相似度越高；
两种方式计算词向量矩阵：CBOW和skip-gram；
中心词和上下文词--定义窗口大小,确定中心词的上下文词,最大化一起出现概率
评估词向量：
1.输出与特定词语相关度高的词语;
2.可视化;--t-SNE算法
3.类比实验：国王-王后=男人-女人--向量相似度
核心目标：两种方式的核心均为迭代出词向量embeddings

CBOW：--上下文词预测中心词--“完形填空”
本质为神经网络，接收上下文词语，将上下文词语转换为最有可能的目标词

a)embeddings层：
1.维度：(N,V),其中N：词表中词语的个数；V：词向量的维度；
2.作用：将输入词语转换为词向量
3.输入词->one-hot（维度：1xN）->embeddings层(维度：NxV)->词向量(维度：1xV)
4.每个中心词的上下文包含多个词语，这些词语均会被输入embedding层并转换为词向量；对这些输出向量求和并求平均--embedding输出是一个将语意信息平均的向量v；

b)线性层：
1.位于embedding层后(所以输入维度为：1xV)，不设置激活函数；
2.权重矩阵维度：(V,N)，N，V含义同embedding层
3.输出维度：1xN；
4.后接softmax函数，最终计算出一个最有可能的输出词，即预测目标词
![[Pasted image 20240916205024.png]]


skip-gram：--中心词预测上下文词
本质为神经网络,接收目标词$w_i$,输出为:词汇表中每个词是目标词上下文词的可能性$p(w_1|w_i),p(w_2|w_i)...p(w_n|w_i)$，其中$w_1\sim w_n$是词表中的词；

迭代过程中，调整词向量，使目标词的词向量与其上下文的词向量尽可能的接近，使目标词的词向量与非上下文词的词向量尽可能的远离；

判断两个词向量是否相似，使用向量的点积：
1.向量点积衡量两个向量在同一方向上的强度；
2.点积越大，两个向量越相似，对应的词语的语义越接近；

a) in_embeddings层：--将目标词转换为词向量
维度：(N，embed_size);N为词表的大小，embed_size为词向量的大小
b) out_embeddings层：--用于表示所有上下文词的词向量
维度：(N，embed_size);N为词表的大小，embed_size为词向量的大小

c):forward：
in_vec = in_embedding()---目标词的词向量
out_vecs = out_embedding()---获取词表中全部词语的词向量
scores = matmul(in_vec,out_vecs.T)---目标词和全部词的点积
将点积输入softmax，计算概率分布，表示每个词是输入词上下文词的概率
![[Pasted image 20240916212715.png]]

词表中的词与目标词：
1.上下文词：正样本，标记为1
2.非上下文词：负样本，标记为0



补充：
词向量/词嵌入(word embedding)：
1.嵌入矩阵的维度:(词汇表中词语个数，词向量的维度)
2.one-hot词矩阵E x 嵌入矩阵E = 结果矩阵(即目标句子的嵌入向量)
3.one-hot编码不通用，嵌入矩阵通用；同一份词向量可以用在不同NLP中

余弦相似度：
$sim(u,v)=\frac{uv}{\vert\vert u \vert\vert_2 \vert\vert v \vert\vert_2}$，其中二范数即：$\vert\vert x \vert\vert_2=\sqrt{x_1^2+x_2^2+..+x_n^2}$
