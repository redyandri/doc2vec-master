from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from flask import request
import flask
from victorinox import victorinox
import os
import logging
from PIL import Image
import cv2
import tensorflow as tf
import jsonify
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
import sys
import time

app = Flask(__name__,static_url_path='')
api = Api(app)
tool=victorinox()
tfidf_vectors=r"data/tfidf_group_vectors.csv"
tfidf_model=r"data/tfidf_group_model.pkl"
df=pd.read_csv(tfidf_vectors,sep=";",header=None)
transformer = TfidfTransformer()
loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open(tfidf_model, "rb")))
t1=time.time()
knn = KNeighborsClassifier(n_neighbors=len(df))
knn.fit(df.iloc[:,:-1], df.iloc[:,-1])
print("train KNN DONE in %f"%(time.time()-t1),file=sys.stdout)

@app.route('/user_by_competence', methods = ['GET'])
def user_by_competence():
    q = request.args.get("q")
    q = tool.preprocess_sentence(q)
    qv = transformer.fit_transform(loaded_vec.fit_transform([q])).toarray()[0].tolist()
    (distances, indices) = knn.kneighbors([qv], n_neighbors=5)
    indices = indices.tolist()[0]
    res = df.iloc[indices, -1]

    return res.to_json(orient='records')


@app.route('/')
def hello_world():
    return 'Hello, World!'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)