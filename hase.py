from nltk.tokenize import RegexpTokenizer
import pandas as pd
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import tokenize
from gensim import utils
import numpy as np


# tokenizer = RegexpTokenizer(r'\w+')
# print(tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!'))

# s=[["1","2","3"],["why","cloud","gloomy"],["why","sun","shady"]]
# df=pd.DataFrame(s,columns=["1","2","3"],index=None)
# print(list(df["2"]))

# documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
# print(documents)

# class MyIter(object):
#     path=""
#     def __init__(self,fp):
#         self.path=fp
#
#     def __iter__(self):
#         # path = datapath(self.path)
#         with utils.open(self.path, 'r', encoding='utf-8') as fin:
#             for line in fin:
#                yield list(tokenize(line))
#
# dataset_path=r"data\dataset_lower_clean_stem_sentence.csv"
# model_path=r"model\doc2vec100.bin"
# corpus=MyIter(dataset_path)
# print([x for x in corpus.__iter__()])

# line="i love erdbeer guys"
# print(len(line.split()))

dataset_path=r"data/dataset_lower_clean_stem_sentence.csv"
counts=[]
with open(dataset_path,"r") as f:
    lines= f.readlines()
    for line in lines:
        counts.append(len(line.split()))
print(np.mean(counts))


