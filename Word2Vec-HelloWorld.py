#!/usr/bin/env python
# encoding: utf-8

import time
import logging

import jieba
import jieba.analyse
from gensim.models import word2vec, KeyedVectors
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import logging.config
# 需要安装 pyymal 库

logger = logging.getLogger(__name__)

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s %(funcName)s %(levelname)s %(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S',
                    filename='Word2Vec.log',
                    filemode='w')

# API: https://radimrehurek.com/gensim/models/word2vec.html

def preHandel(path):
    st = time.time()
    num = 0

    sentences = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line.strip() != "":
                # `[^\w\s]` 匹配除了字母、数字和空格之外的所有字符
                content = re.sub('[^\w\s]', '', line.strip())
                # jieba 分词获取词语序列
                content_seq = list(jieba.cut(content))
                sentences.append(content_seq)

                num += 1

    end = time.time()
    print("PreHandel End Num:%s Cost:%ss" % (num, (end - st)))
    return sentences


def getSimilarSeq(key, model, top=10):
    print("Top For %s ======================" % key)
    sims = model.wv.most_similar(key, topn=top)
    for i in sims:
        print(i)
    print("End Sim For %s ======================" % key)


if __name__ == "__main__":

    logger.warning("This is warning from Nathan")
    # 1.数据预处理
    path = "Corpus/all.txt"
    sentences = preHandel(path)

    """
        2.模型训练
        - sentences: 可迭代的语句列表，较大的语料库可以考虑从磁盘/IO的形式传输
        - vector_size: 单词向量的维数
        - window: 句子中当前单词与预测单词的最大距离
        - min_count: 忽略总频率低于此值的所有单词
        - workers: 使用多个 worker 线程训练模型
        - sg: 训练算法，1-> skip-gram 否则 -> CBOW
        - hs: 1 -> 分层 softmax 方法，否则 -> 负采样
        - negative: >0 则使用负采样，通常推荐距离为 [5-20]，如果设置为0则不适用负采样
        - alpha: 初始学习率
        - min_alpha: 随着训练进行，学习率将线性下降至 min_alpha
        - max_vocab_size: 词库限制，每 1000w 个字类型大约需要1GB的 RAM
        - sample: 配置较高频率的单词随机下采样的阈值，生效范围 (0,1e-5)
        - epoch: 迭代次数
    """
    w2v = word2vec.Word2Vec(sentences, hs=1, sg=1, min_count=1, window=3, vector_size=300, workers=4)

    print(w2v.wv["孔明"])
    # 3.寻找最优
    getSimilarSeq("孔明", w2v)
    getSimilarSeq("周瑜", w2v)

    # 4.寻找不匹配
    print(w2v.wv.doesnt_match("孔明 张飞 关公 先主 张辽".split()))

    # 5.快速加载 Store just the words + their trained embeddings. 使用内存映射进行加载=只读，跨进程共享。
    # word_vectors = w2v.wv
    # word_vectors.save("./word2vec.wordvectors")
    # wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')
    # vector = wv['赵云']
    # print(vector)

    # 6.模型保存于重训练
    # w2v.save("./model/w2v.model")
    # reloadW2V = Word2Vec.load('./model/w2v.model')
    # new_sentences = preHandel("./retrain.txt")
    #
    # reloadW2V.train(new_sentences, total_examples=reloadW2V.corpus_total_words, epochs=5)
    #
    # getSimilarSeq("孔明", reloadW2V)
    # getSimilarSeq("周瑜", reloadW2V)

    # 7.数据可视化
    key_list = []
    emb_list = []
    candidate_list = ["诸葛亮", "刘备", "关羽", "张飞", "周瑜", "鲁肃", "吕蒙", "陆逊"]

    for k in candidate_list:
        emb_list.append(w2v.wv[k])
        key_list.append(k)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    pca = PCA(n_components=2)
    compress_emb = pca.fit_transform(np.array(emb_list))

    print(compress_emb)

    candidate_x = [compress_emb[index, 0] for index in range(len(candidate_list))]
    candidate_y = [compress_emb[index, 1] for index in range(len(candidate_list))]

    plt.scatter(candidate_x, candidate_y, s=10)
    for x, y, key in zip(candidate_x, candidate_y, key_list):
        plt.text(x, y, key, ha='left', rotation=0, c='black', fontsize=8)
    plt.title("PCA")
    plt.rcParams['axes.unicode_minus']=False
    plt.show()

