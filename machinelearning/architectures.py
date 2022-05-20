from machinelearning  import helpers 
import pandas as pd
import numpy as np
import config
import tensorflow as tf
from tensorflow import keras

def prediccion_con_randomforest(sentencia):
    prediction_text = sentencia.lower()
    prediction_text = helpers.remover_html(prediction_text)
    prediction_text = helpers.remover_caracteres_especiales(prediction_text)
    prediction_text = helpers.mantener_caracteres_alfabeticos(prediction_text)
    prediction_text = helpers.remover_stopwords(prediction_text,config.stop_words)
    prediction_text = helpers.lemmatizar_con_postag(prediction_text)
    data=[[prediction_text]]
    df = pd.DataFrame(data, columns = ['plot'])
    x_dtm= config.vectorizer.transform(df['plot'])
    print(x_dtm)
    yPred=config.model_rf.predict(x_dtm)
    y_labels_pred= config.multilabelbin.inverse_transform(yPred)
    y_labels_pred= list(y_labels_pred)
    return (prediction_text,y_labels_pred)

def prediccion_con_embedding_nn(sentencia):
    prediction_text = sentencia.lower()
    prediction_text = helpers.remover_html(prediction_text)
    prediction_text = helpers.remover_caracteres_especiales(prediction_text)
    prediction_text = helpers.mantener_caracteres_alfabeticos(prediction_text)
    prediction_text = helpers.remover_stopwords(prediction_text,config.stop_words)
    
    sequencias= config.tokenizer.texts_to_sequences(prediction_text)
    secuencias=  tf.keras.preprocessing.sequence.pad_sequences(sequences= sequencias,padding='post')
    out = config.model_nn.predict(secuencias)
    out = np.array(out)
    print(out.shape)
    y_pred = np.zeros(out.shape)
    y_pred[out>0.5]=1
    y_pred = np.array(y_pred)
    y_labels_pred=config.multilabelbin.inverse_transform(y_pred)
    return (y_labels_pred)  
    