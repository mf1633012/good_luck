import tensorflow as tf
import numpy as np
import os, argparse, time, random
from model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity
from data import read_corpus, read_dictionary, tag2label, random_embedding
from sklearn import preprocessing
from gensim import corpora, models, similarities

## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory


## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=False, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='pretrain', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=150, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1552059104', help='model for test and demo')
parser.add_argument('--lambda1', type=float, default=0, help='first weight')
parser.add_argument('--lambda2', type=float, default=0, help='second weight')
args = parser.parse_args()


## get char embeddings
word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    embedding_path = 'pretrain_embedding.npy'
    #embeddings = np.array(np.load(embedding_path))
    A=np.load(embedding_path)
#    embeddings_mat = np.zeros((len(word2id), args.embedding_dim),dtype=np.float32)
    embeddings_mat = np.random.uniform(-0.25, 0.25, (len(word2id), args.embedding_dim))
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
    terms_topic=np.zeros([len(lda.id2word),50])
    for i in range(len(lda.id2word)):
        result=lda.get_term_topics(i,0)
        for j in result:
            terms_topic[i][j[0]]=j[1]
#    for i in range(len(word2id)):
#        embeddings_mat[i] = A[list(word2id.keys())[list(word2id.values()).index(i)]]
#        embeddings_mat[i] = A[list(word2id.keys())[0]]
    sum=0
    for i in terms_topic:
        for j in range(terms_topic.shape[1]):
            sum+=i[j]
    average_lda=sum/(terms_topic.shape[0]*terms_topic.shape[1])        
    max_lda=max(terms_topic.reshape(-1,1))[0]
    min_lda=min(terms_topic.reshape(-1,1))[0]

      
    k=0
    for i in word2id:
        if i in A:
            embeddings_mat[k,0:100]=A[i]
        else:
            print(i)
        k=k+1
    
    word_matrix=embeddings_mat[:,0:100]
    
    sum=0
    for i in word_matrix:
        for j in range(word_matrix.shape[1]):
            sum+=i[j]
    average_w2v=sum/(word_matrix.shape[0]*word_matrix.shape[1])        
    max_w2v=max(word_matrix.reshape(-1,1))[0]
    min_w2v=min(word_matrix.reshape(-1,1))[0]
    
    u=(average_w2v-min_w2v)/(average_lda-min_lda)
    
    
    #放缩
    sum=0
    for i in range(terms_topic.shape[0]):
        for j in range(terms_topic.shape[1]):
            terms_topic[i][j]=terms_topic[i][j]*u+min_w2v
            sum+=terms_topic[i][j]
    average_lda=sum/(terms_topic.shape[0]*terms_topic.shape[1])  
    
    k=0    
    for i in word2id:
        if i in lda.id2word.token2id:
            wordId=lda.id2word.token2id[i]
            wordVec=terms_topic[wordId]
            embeddings_mat[k,100:150]=wordVec
        else:
            print(i)
        k=k+1
    embeddings=embeddings_mat

## read corpus and get training data
if args.mode != 'demo':
    train_path = os.path.join('.', args.train_data, 'train_data')
    test_path = os.path.join('.', args.test_data, 'test_data')
    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path); test_size = len(test_data)


## paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join('.', args.train_data+"_save", timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")#cause the error?
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))


## training model
if args.mode == 'train':
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()

    ## hyperparameters-tuning, split train/dev
    # dev_data = train_data[:5000]; dev_size = len(dev_data)
    # train_data = train_data[5000:]; train_size = len(train_data)
    # print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
    # model.train(train=train_data, dev=dev_data)

    ## train model on the whole training data
    print("train data: {}".format(len(train_data)))
    model.train(train=train_data, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena

## testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("test data: {}".format(test_size))
    model.test(test_data)

## demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        while(1):
            print('文本输入:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                demo_sent = list(demo_sent.strip())
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                tag = model.demo_one(sess, demo_data)
                PER, LOC, ORG = get_entity(tag, demo_sent)
                print('人名实体: {}\n地名实体: {}\n组织名实体: {}'.format(PER, LOC, ORG))
