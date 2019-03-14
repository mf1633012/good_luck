#coding:utf8


from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence


def loadModel(filepath):
    #model = KeyedVectors().load_word2vec_format(filepath,binary=False)
    model = Word2Vec.load(filepath)
    return model

if __name__=="__main__":
    #train("D:/wiki_space.txt")
    ''''''
    model = loadModel("WordEmbedding_chs_100d.vec")
    for v,s in model.most_similar("育",topn=15):
        print(v,s)
    