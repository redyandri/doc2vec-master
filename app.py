import pymssql
from flask import Flask
from flask import render_template
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from flask import request
import flask
from victorinox import victorinox
import os
import logging
from PIL import Image
#import cv2
#import tensorflow as tf
import jsonify
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
import sys
import time
import json
from flask import Response

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
    dbConn = pymssql.connect('10.242.77.202', 'ecorp', 'Pusintek2016##', "nadine")
    oldCursor = dbConn.cursor()
    dbConn.commit()
    data = list(res)
    response = []
    nips = []
    for x in data:
        nips.append(x.split('_')[0])
    query = 'select nama, nip18, ref_unit.nama_organisasi from ref_user inner join ref_unit on ref_user.id_organisasi=ref_unit.id_organisasi where nip18 in (%s)'%(','.join(("'{0}'".format(w) for w in nips)))
    print(query)
    oldCursor.execute(query)
    result = []
    for x in oldCursor.fetchall():
        result.append({'nama': x[0], 'nip': x[1], 'unit': x[2]})
    js = json.dumps(result)

    resp = Response(js, status=200, mimetype='application/json')
    return json.dumps(result)


@app.route('/')
def hello_world():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
