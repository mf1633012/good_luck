# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 18:13:54 2019

@author: Gu Yi
"""

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
# 训练的语料
def loadModel(filepath):
    #model = KeyedVectors().load_word2vec_format(filepath,binary=False)
    model = Word2Vec.load(filepath)
    return model
# 利用语料训练模型
model = loadModel("WordEmbedding_chs_100d.vec")

# 基于2d PCA拟合数据
#X = model[model.wv.vocab]
#pca = PCA(n_components=2)
#result = pca.fit_transform(X)
# 可视化展示

words = ['英','美','法','德','请','求','帮','助','我','是','顾','溢','这','样','子','的','啊','苏','鲁','豫','湘','鄂','桂','粤','玩','万','弯','尬','天','一','二','三','四']
X=model[words]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

pyplot.rcParams['savefig.dpi'] = 100
pyplot.rcParams['figure.dpi'] = 100
for i, word in enumerate(words):
    pyplot.scatter(result[i, 0], result[i, 1],marker = 'o', s = 20,color='blue')
    pyplot.annotate(i,xy=(result[i, 0], result[i, 1]),)
    print(i,word)
pyplot.show()
