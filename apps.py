from ast import Str
import numpy as np
from flask import Flask, render_template, request
import pickle
import sys
import logging

app = Flask(__name__)  # Initialize the flask App

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

model = pickle.load(open('model.pkl', 'rb'))  # loading the trained model


@app.route('/')  # Homepage
def home():
    return render_template('output.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
   
    # retrieving values from form
    init_features =[str(x) for x in request.form.values()]
    #init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]

    prediction = model.predict(final_features)  # making prediction

    return render_template('output.html',
                           prediction_text='Predicted Class: {}'.format(prediction))  # rendering the predicted result


if __name__ == "__main__":
    app.run(debug=True)