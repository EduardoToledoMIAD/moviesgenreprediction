from machinelearning  import helpers 
import pandas as pd
import numpy as np
import config


def prediccion_con_randomforest(sentencia):
    prediction_text = sentencia.lower()
    prediction_text = helpers.remover_html(prediction_text)
    prediction_text = helpers.remover_caracteres_especiales(prediction_text)
    prediction_text = helpers.mantener_caracteres_alfabeticos(prediction_text)
    prediction_text = helpers.remover_stopwords(prediction_text)
    prediction_text = helpers.lemmatizar_con_postag(prediction_text)
    data=[[prediction_text]]
    df = pd.DataFrame(data, columns = ['plot'])
    x_dtm= config.vectorizer.transform(df['plot'])
    yPred=config.model_rf.predict(x_dtm)
    y_labels_pred= config.multilabelbin.inverse_transform(yPred)
    y_labels_pred= list(y_labels_pred)
    return (prediction_text,y_labels_pred)




    