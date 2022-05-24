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
import  machinelearning.helpers as helpers
import machinelearning.architectures as ML
import joblib
import config
from models.schemas import BatchSchema


class MoviesGenresPrediction(Resource):
    def _features_selection(self, df:pd.DataFrame):
        features=['Year', 'Mileage', 'Make', 'Model', 'State']
        df= df[features]
        return df

    def _text_preprocessing(self, json):
        df = pd.DataFrame(json,columns=['plot'])
        df['plot'] = df['plot'].str.lower()
        df['plot'] = df['plot'].apply(lambda x: helpers.remover_html(x))
        df['plot'] = df['plot'].apply(lambda x:helpers.remover_caracteres_especiales(x))
        df['plot'] = df['plot'].apply(lambda x:helpers.mantener_caracteres_alfabeticos(x))
        df['plot'] = df['plot'].apply(lambda x: helpers.remover_stopwords(x))
        df['plot'] = df['plot'].apply(lambda x: helpers.lemmatizar_con_postag(x))
        return df
    
    def post(self):
        global model
        data= request.get_json()
        try:
            json =BatchSchema().load(data)
            df=self._text_preprocessing(json['batch'])
            x_dtm= config.vectorizer.transform(df['plot'])
            yPred=config.model_rf.predict(x_dtm)
            y_labels_pred= config.multilabelbin.inverse_transform(yPred)
           # y_labels_pred= list(y_labels_pred)
            res = [','.join(i) for i in y_labels_pred]
            return {'predictions': str(res)},200
          
        except ValidationError as err:
            print(err.messages)
            return err.messages, 400
        except Exception as e:
            traceback.print_exc()
            return traceback.print_exc(),500


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
    tipo_modelo= str(request.form['optratio'])
    if tipo_modelo == 'RF':
        prediction_text,y_labels_pred=ML.prediccion_con_randomforest(sentencia)
    elif tipo_modelo == 'NN':
        prediction_text,y_labels_pred=ML.prediccion_con_embedding_nn(sentencia)
    else:
        prediction_text,y_labels_pred=ML.prediccion_con_embedding_nn(sentencia)
    return render_template('index.html', id='predict', prediction_text=prediction_text, result=y_labels_pred)        

@app.route('/api/doc',methods=['GET'])
def api_documentation():
    return render_template('documentation.html')

api.add_resource(MoviesGenresPrediction, '/api/moviesgenres-prediction')  
if __name__ == '__main__':
    app.run()