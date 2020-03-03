from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import tokenize
from gensim import utils

class MyIter(object):
    path=""
    def __init__(self,fp):
        self.path=fp

    def __iter__(self):
        # path = datapath(self.path)
        with utils.open(self.path, 'r', encoding='utf-8') as fin:
            for line in fin:
               yield list(tokenize(line))

dataset_path=r"data\dataset_lower_clean_stem_sentence.csv"
model_path=r"model\doc2vec100.bin"
corpus=MyIter(dataset_path)
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
d2v_model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4)
d2v_model.build_vocab(documents)
d2v_model.train(documents,total_words=d2v_model.corpus_count,epochs=d2v_model.epochs)
d2v_model.save(model_path)

model_path=r"model\doc2vec100.bin"
d2v_model = Doc2Vec.load(model_path)
print(d2v_model.wv.most_similar(["naskah","dinas"]))