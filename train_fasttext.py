from victorinox import victorinox
from gensim.models import FastText
from gensim.test.utils import datapath
from gensim.test.utils import get_tmpfile
from gensim.utils import tokenize
from gensim import utils
import logging



#enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MyIter(object):
    path=""
    def __init__(self,fp):
        self.path=fp

    def __iter__(self):
        # path = datapath(self.path)
        with utils.open(self.path, 'r', encoding='utf-8') as fin:
            for line in fin:
               yield list(tokenize(line))

#https://radimrehurek.com/gensim/models/fasttext.html
# vector_size = 100
# window_size = 5
# min_count = 1
# sampling_threshold = 1e-5
# negative_size = 5
# train_epoch = 100
# dm = 0 #0 = dbow; 1 = dmpv
# worker_count = 1
# sg=0
# dataset_path=r"data\dataset_lower_clean_stem_sentence.csv"
# corpus_file = datapath(dataset_path)
# model_path=r"model\fasttext100.bin"
# pretrained=r"model\cc.id.300.bin\cc.id.300.bin"
# pretrained_vec=r"model\cc.id.300.vec\cc.id.300.vec"
# corpus=MyIter(dataset_path)
# fasttext_model=FastText(corpus,
#                         size=vector_size,
#                         window=window_size,
#                         min_count=min_count,
#                         sample=sampling_threshold,
#                         workers=worker_count,
#                         hs=0,
#                         sg=sg,
#                         #dm=dm,
#                         negative=negative_size,
#                         #dbow_words=1,
#                         #dm_concat=1,
#                         #pretrained_emb=pretrained_vec,
#                         iter=train_epoch)
# fasttext_model.save(model_path)
#
#
# model_path=r"model\fasttext100.bin"
# fasttext_model = FastText.load(model_path)
# sim=fasttext_model.wv.most_similar(['nosql', 'mongodb'])
# print(sim)


vector_size = 100
window_size = 5
min_count = 5
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 0 #0 = dbow; 1 = dmpv
worker_count = 1
sg=0
dataset_path=r"data/dataset_lower_clean_stem_sentence.csv"
corpus_file = datapath(dataset_path)
model_path=r"model/fasttext100/fasttext100_retrain.bin"
pretrained=r"model/fasttext100/fasttext100.bin"
pretrained_vec=r"model\cc.id.300.vec\cc.id.300.vec"
corpus=MyIter(dataset_path)
fasttext_model=FastText(corpus,
                        size=vector_size,
                        window=window_size,
                        min_count=min_count,
                        sample=sampling_threshold,
                        workers=worker_count,
                        hs=0,
                        sg=sg,
                        #dm=dm,
                        negative=negative_size,
                        #dbow_words=1,
                        #dm_concat=1,
                        #pretrained_emb=pretrained_vec,
                        iter=train_epoch)
fasttext_model.save(model_path)


model_path=r"model\fasttext100.bin"
fasttext_model = FastText.load(model_path)
sim=fasttext_model.wv.most_similar(['nosql', 'mongodb'])
print(sim)
