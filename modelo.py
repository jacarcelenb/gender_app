from keras.models import load_model
from matplotlib.pyplot import axis
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn import decomposition
import pandas as pd

model_gender = load_model("model.h5")

admissions = pd.read_csv('minmax.csv')
datos = admissions.values
datos = np.array(datos , "float32")

def predecir(lista):
    lista = np.array(lista , "float32")
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    datos= np.concatenate((datos,[lista]) , axis=0)
    datos = min_max_scaler.fit_transform(datos)
    pca = decomposition.PCA(n_components=2)
    datos = pca.fit_transform(datos)
    datos = np.array([datos[-1]])
    prediction = model_gender.predict([datos]).argmax(axis= 1)
    print("Prediccion ")
    print(prediction)
    return prediction

lista = [1,11.8 ,6.1,1,0,1,1]
predecir(lista)
    


