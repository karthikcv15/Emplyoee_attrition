# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 21:16:25 2021

@author: Karthik C V
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model2=pickle.load(open('model2.pkl','rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model2.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Attrition is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)