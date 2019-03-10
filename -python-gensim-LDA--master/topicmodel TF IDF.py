import jieba, os
from gensim import corpora, models, similarities

train_set = []

walk = os.walk('output1')
for root, dirs, files in walk:
    for name in files:
        f = open(os.path.join(root, name), 'r',encoding='gbk')
        raw = f.read()
        #word_list = list(jieba.cut(raw, cut_all = False))

        word_list=[]
        for word in raw:
            if(word!=' 'and word!='\n'):
                word_list.append(word)

        train_set.append(word_list)


dic = corpora.Dictionary(train_set)
corpus = [dic.doc2bow(text) for text in train_set]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lda = models.LdaModel(corpus_tfidf, id2word = dic, num_topics = 50)
corpus_lda = lda[corpus_tfidf]

for topic in lda.print_topics(num_words = 10):
    termNumber = topic[0]
    print(topic[0], ':', sep='')
    listOfTerms = topic[1].split('+')
    for term in listOfTerms:
        listItems = term.split('*')
        print('  ', listItems[1], '(', listItems[0], ')', sep='')
        
        
#terms_topics matrix        
import numpy as np
terms_topic=np.zeros([len(lda.id2word),50])
for i in range(len(lda.id2word)):
    result=lda.get_term_topics(i,0)
    for j in result:
        terms_topic[i][j[0]]=j[1]*10000




"""   
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
    for v,s in model.most_similar("шоо",topn=15):
        print(v,s)
"""  


wordId=lda.id2word.token2id['育']
wordVec=terms_topic[wordId]
distance=[]
for i in range(len(lda.id2word)):
    termVec=terms_topic[i]
    dist=np.linalg.norm(wordVec - termVec)
    distance.append([i,dist])

def takeSecond(elem):
    return elem[1]

distance.sort(key=takeSecond)

for i in range(15):
    print(lda.id2word.id2token[distance[i][0]]," d=",distance[i][1])
