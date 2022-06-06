# -*- coding: utf-8 -*-

import flask
import pandas as pd
import numpy as np
import tensorflow as tf

from flask import Flask, jsonify, request, render_template
from keras.models import load_model

app = Flask(__name__)

global graph
graph = tf.compat.v1.get_default_graph()


#입력폼 html
@app.route('/smartcar/predict')
def form():
    return render_template('form.html')
#enddef

@app.route('/smartcar/predict/result',  methods=["GET","POST"])
def predict():
    data = {"success": False}

    if (request.method == 'POST'):
        params = request.form
    #endif

    if (params == None):
        params = flask.request.args
    #endif

    # if parameters are found, return a prediction
    if (params != None):
        x=pd.DataFrame.from_dict(params, orient='index').transpose()

        with graph.as_default():
            model = load_model('/root/IPULearning/SmartCar/smartcar_dnn_model.h5')
            data["prediction"] = str(model.predict(x).argmax())
            data["success"] = True
        #endwith
    #endif

    # return a response in json format
    # return render_template("result.html", result = jsonify(data)) #결과 출력 html
    return flask.jsonify(data)
#enddef

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9001)
#endif
