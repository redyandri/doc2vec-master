from gensim.models import Word2Vec
from gensim.utils import tokenize
from gensim import utils
from gensim.test.utils import datapath
from gensim.utils import tokenize
from gensim import utils
import logging
from gensim.models import KeyedVectors
import gensim.downloader as api
from victorinox import victorinox

#enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MyIter(object):
    path=""
    def __init__(self,fp):
        self.path=fp

    def __iter__(self):
        # path = datapath(self.path)
        with open(self.path, 'r', encoding='utf-8') as fin:
            for line in fin:
               yield list(tokenize(line))

# #https://radimrehurek.com/gensim/models/doc2vec.html
# vector_size = 300
# window_size = 4
# min_count = 1
# sampling_threshold = 1e-5
# negative_size = 5
# train_epoch = 100
# dm = 0 #0 = dbow; 1 = dmpv
# sg=1        #CBOW
# worker_count = 1
# dataset_path=r"data/dataset_lower_clean_stem_sentence.csv"
# corpus_file = datapath(dataset_path)
# model_path=r"model/word2vec100_cbow.bin"
# corpus=MyIter(dataset_path)
# word2vec_model=Word2Vec(corpus,
#                         size=vector_size,
#                         window=window_size,
#                         min_count=min_count,
#                         sample=sampling_threshold,
#                         workers=worker_count,
#                         hs=0,
#                         #dm=dm,
#                         negative=negative_size,
#                         #dbow_words=1,
#                         #dm_concat=1,
#                         #pretrained_emb=pretrained_vec,
#                         sg=sg,
#                         iter=train_epoch)
# word2vec_model.save(model_path)

# model_path=r"model/word2vec100_cbow.bin"
# w2v_model = Word2Vec.load(model_path)
# print(w2v_model.wv.most_similar("pegawai"))
# print(w2v_model.wv.most_similar("gaji"))


# vector_size = 300
# window_size = 4
# min_count = 1
# sampling_threshold = 1e-5
# negative_size = 5
# train_epoch = 100
# dm = 0 #0 = dbow; 1 = dmpv
# sg=0      #SKIP ThOUGHT
# worker_count = 1
# dataset_path=r"data/dataset_lower_clean_stem_sentence.csv"
# corpus_file = datapath(dataset_path)
# model_path=r"model/word2vec300_skipthought.bin"
# corpus=MyIter(dataset_path)
# word2vec_model=Word2Vec(corpus,
#                         size=vector_size,
#                         window=window_size,
#                         min_count=min_count,
#                         sample=sampling_threshold,
#                         workers=worker_count,
#                         hs=0,
#                         #dm=dm,
#                         negative=negative_size,
#                         #dbow_words=1,
#                         #dm_concat=1,
#                         #pretrained_emb=pretrained_vec,
#                         sg=sg,
#                         iter=train_epoch)
# word2vec_model.save(model_path)
#
# model_path=r"model/word2vec300_skipthought.bin"
# w2v_model = Word2Vec.load(model_path)
# print(w2v_model.wv.most_similar("database"))


idwiki_word2vec_model=r"model/idwiki_word2vec_300/idwiki_word2vec_300.model"
idwiki_word2vec_model_retrain=r"model/idwiki_word2vec_300/idwiki_word2vec_300_retrain.model"
dataset_path=r"data/dataset_lower_clean_stem_sentence.csv"
corpus=MyIter(dataset_path)
vector_size = 300
window_size = 4
min_count = 1
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 0 #0 = dbow; 1 = dmpv
sg=0      #SKIP ThOUGHT
worker_count = 1
word2vec_model=Word2Vec.load(idwiki_word2vec_model)
# word2vec_model=api.load(idwiki_word2vec_model)
# word2vec_model.build_vocab(sentences=corpus,update=True)
# word2vec_model.train(corpus,
#                      total_examples=word2vec_model.corpus_count,
#                      epochs=train_epoch)
# word2vec_model.save(idwiki_word2vec_model_retrain)
word2vec_model2=Word2Vec.load(idwiki_word2vec_model_retrain)
# print(word2vec_model.wv.most_similar("yang"))
# print(word2vec_model2.wv.most_similar("yang"))
tool=victorinox()
s1="kemampuan analisa sql server"
s2="analisa jaringan komputer"
s3="pengolahan database"
s1_emb=tool.document_vector(word2vec_model2,s1)
s2_emb=tool.document_vector(word2vec_model2,s2)
s3_emb=tool.document_vector(word2vec_model2,s3)
print(tool.measure_similarity(s1_emb,s1_emb))
print(tool.measure_similarity(s1_emb,s2_emb))
print(tool.measure_similarity(s1_emb,s3_emb))
print(tool.measure_similarity(s2_emb,s3_emb))
print("." in word2vec_model.wv.vocab)