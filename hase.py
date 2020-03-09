from nltk.tokenize import RegexpTokenizer
import pandas as pd
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import tokenize
from gensim import utils
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

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

# dataset_path=r"data/dataset_lower_clean_stem_sentence.csv"
# counts=[]
# with open(dataset_path,"r") as f:
#     lines= f.readlines()
#     for line in lines:
#         counts.append(len(line.split()))
# print(np.mean(counts))

# s="d.i.k.l.a.t. .p.e.r.i.k.s.a. .l.a.n.g.g.a.r. .d.i.s.i.p.l.i.n. .p.e.g.a.w.a.i. .i.n.d.e.p.e.n.d.e.n.t. .s.t.u.d.y"
# regex = re.compile(r"([a-z].)+", re.IGNORECASE)
# regex2 = re.compile(r"\d{18},", re.IGNORECASE)
# nip = regex.findall(s)#[0]
# print(nip)
# #line = regex.sub(nip + ";", line)

# from sklearn.feature_extraction.text import TfidfVectorizer
# corpus = [
# 'This is the first document.',
# 'This document is the second document.',
# 'And this is the third one.',
# 'Is this the first document?' ]
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit(corpus)
# print(vectorizer.get_feature_names())
# print(X.transform(["this is the one"]).toarray()[0])
# print(X.transform(["this is the one i choose"]).toarray()[0])

# import pickle
#
# a = {'hello': 'world'}
#
# with open('filename.pickle', 'wb') as handle:
#     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)
#
# print(a == b)


# corpus = np.array(["aaa bbb ccc", "aaa bbb ddd"])
# vectorizer = CountVectorizer(decode_error="replace")
# vec_train = vectorizer.fit_transform(corpus)
# pickle.dump(vectorizer.vocabulary_,open("feature.pkl","wb"))
#
# #Load it later
# transformer = TfidfTransformer()
# loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
# tfidf = transformer.fit_transform(loaded_vec.fit_transform(np.array(["aaa ccc eee"])))
# print(tfidf.toarray()[0])

# d={"nip_name":["123_andri","456_ana"],"kompetensi":["ai","bio"]}
# l=["123"]
# df=pd.DataFrame(d)
# ar=[x for x in df.nip_name]
# print(ar)


X_tr=[[1,2,3],[4,5,6]]
y_train=[1,2]
lr = SGDClassifier()#SGDClassifier(loss='hinge',alpha=best_alpha,class_weight='balanced')
clf =lr.fit(X_tr, y_train)
calibrator = CalibratedClassifierCV(clf, cv='prefit')
model=calibrator.fit(X_tr, y_train)
yy=clf.predict_proba([[1,2,3]])
y_train_pred = model.predict_proba(X_tr)
y_test_pred = model.predict_proba([[1,2,3]])
print(yy)
print(y_train_pred)
print(y_test_pred)


