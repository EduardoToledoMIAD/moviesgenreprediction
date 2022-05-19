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
import helpers
import joblib

print( nltk.__version__)

app= Flask(__name__, template_folder="templates")

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

# Definición API Flask 

api = Api(app)

vectorizer =  joblib.load('vectorizer_tfid.pkl')
multilabelbin= joblib.load('multilabelbinarizer.pkl')
stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten',
                   'may','also','across','among','beside','however','yet','within',
                   'and','or'])
data = [['tom$ zero don him', 10], ['tagging nick<> ', 15], ['juli.. are', 14]]
df = pd.DataFrame(data, columns = ['plot', 'Id'])
df['plot_trans'] = df['plot'].str.lower()
df['plot_trans'] = df['plot_trans'].apply(lambda sentencia:helpers.remover_stopwords(sentencia,stop_words))
df['plot_trans'] = df['plot_trans'].apply(lambda sentencia: helpers.remover_html(sentencia))
df['plot_trans'] = df['plot_trans'].apply(lambda sentencia:helpers.remover_caracteres_especiales(sentencia))
df['plot_trans'] = df['plot_trans'].apply(lambda sentencia:helpers.mantener_caracteres_alfabeticos(sentencia))
df['plot_trans'] = df['plot_trans'].apply(lambda sentencia:helpers.lemmatizar_con_postag(sentencia))

print(df['plot_trans']) 
@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    global vectorizer
    global multilabelbin
    prediction_text=str(request.form['notes'])
    prediction_text = prediction_text.lower()
    prediction_text = helpers.remover_html(prediction_text)
    prediction_text = helpers.remover_caracteres_especiales(prediction_text)
    prediction_text = helpers.mantener_caracteres_alfabeticos(prediction_text)
    prediction_text = helpers.lemmatizar_con_postag(prediction_text)
    data=[[prediction_text]]
    df = pd.DataFrame(data, columns = ['plot'])
    x_dtm= vectorizer.transform(df['plot'])
    print(x_dtm)
    return render_template('index.html', id='predict', prediction_text=prediction_text)        

@app.route('/api/doc',methods=['GET'])
def api_documentation():
     return render_template('documentation.html')
    
if __name__ == '__main__':
    app.run()