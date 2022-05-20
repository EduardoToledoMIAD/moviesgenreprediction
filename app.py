from base64 import encode
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_restful import   Api,Resource,reqparse
from marshmallow import Schema, fields, validate, ValidationError
import pickle
import joblib
import traceback
import nltk
from nltk.corpus import stopwords
import machinelearning.architectures as ML
import joblib
import config
print( nltk.__version__)

app= Flask(__name__, template_folder="templates")

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

# Definición API Flask 

api = Api(app)

config.vectorizer =  joblib.load('vectorizer_tfid.pkl')
config.multilabelbin= joblib.load('multilabelbinarizer.pkl')
config.model_rf = joblib.load('model_RF.pkl')

#try:
#    config.tokenizer = joblib.load('tokenizer.pkl') 
#    config.model_nn = joblib.load('model_NN.pkl')
    
#except Exception as e:
#    print("Error")


config.stop_words = set(stopwords.words('english'))
config.stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten',
                   'may','also','across','among','beside','however','yet','within',
                   'and','or'])
@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    sentencia=str(request.form['notes'])
    prediction_text,y_labels_pred=ML.prediccion_con_randomforest(sentencia)
    return render_template('index.html', id='predict', prediction_text=prediction_text, result=y_labels_pred)        

@app.route('/api/doc',methods=['GET'])
def api_documentation():
    #y_labels_pred=ML.prediccion_con_embedding_nn("Text: Command ignored: No document in device Escrow")
    return render_template('documentation.html')
    
if __name__ == '__main__':
    app.run()