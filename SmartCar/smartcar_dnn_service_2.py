# -*- coding: utf-8 -*-
import flask
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from keras.models import load_model

app = Flask(__name__)

global graph
graph = tf.compat.v1.get_default_graph()

@app.route('/smartcar/predict',  methods=["GET","POST"])
def predict():
    data = {"success": False}

    params = flask.request.json

    if (params == None):
        params = flask.request.args
    #endif

    # if parameters are found, return a prediction
    if (params != None):
        x=pd.DataFrame.from_dict(params, orient='index').transpose()

        with graph.as_default():
            model = load_model('smartcar_dnn_model.h5')
            data["prediction"] = str(model.predict(x).argmax())
            data["success"] = True
        #endwith
    #endif

    return flask.jsonify(data)
#enddef

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9002)
#endif
